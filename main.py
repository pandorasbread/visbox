import datetime
import json
import pathlib
from pathlib import Path
import ffmpeg
import librosa
from matplotlib import animation
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as pyplot
import numpy as np
import pydub

import drawable
from drawable import AxisType, ShapeType, Drawable
from settings import Settings


def determineFfmpegPath():
    path = Path('.\\ffmpeg.exe')

    if (path.is_file()):
        return '.\\ffmpeg.exe'

    return 'ffmpeg'

pyplot.rcParams['animation.ffmpeg_path'] = determineFfmpegPath()


def loadFromFile():
    filepath = settings.audio_file_path
    filetype = pathlib.Path(filepath).suffix
    filename = pathlib.Path(filepath).stem
    print('File type is ' + filetype)
    if (filetype == '.mp3'):
        print('Converting to .wav...')
        audiodata = pydub.AudioSegment.from_mp3(settings.audio_file_path)
        newFilePath = './' + filename + '.wav'
        print('Saving new .wav file...')
        audiodata.export(newFilePath, format='wav')
        print('File saved.')
        filepath = newFilePath

    print('Loading data from audio file.')
    return librosa.load(filepath)


def generate_visualizer():
    timeseries, samplerate = loadFromFile()
    spectrogram, freqaxis, maxbarvalue = getSpectrogramAndFrequencyAxisAndMaxDb(timeseries, samplerate)
    fig = configurePlot()
    createDrawables(fig, spectrogram, freqaxis, maxbarvalue)
    makeVideo(fig, freqaxis, len(spectrogram))


def getSpectrogramAndFrequencyAxisAndMaxDb(timeseries, samplerate):
    # skip parameter multiplier is one over the framerate. we can use it as a multiplier to pass less parameters around.
    skipParam = int(samplerate * (1 / settings.framerate))

    shortTimeFourierTransform = np.abs(librosa.stft(timeseries, hop_length=skipParam, n_fft=2048 * 4))
    spectrogram = librosa.amplitude_to_db(shortTimeFourierTransform, ref=np.max)

    freqaxis = librosa.core.fft_frequencies(samplerate, 2048 * 4)

    freqaxisTemp = [x for x in freqaxis if x <= settings.max_frequency]

    freqaxis = freqaxisTemp
    spectrogram = spectrogram[0:len(freqaxis)] + 80

    maxbarvalue = 0.1;

    newspec = []
    newfreq = []
    vals = np.arange(0, len(spectrogram) - int(len(spectrogram) / settings.number_of_points),
                     int(len(spectrogram) / settings.number_of_points))
    for i in vals:
        newspec.append(np.array(spectrogram[i]))
        for j in spectrogram[i]:
            if (j > maxbarvalue):
                maxbarvalue = j
        newfreq.append(freqaxis[i])

    freqaxis = np.array(arrange_freq_axis_for_chart_type(newfreq))
    spectrogram = np.array(newspec)

    spectrogram = spectrogram.T
    if (settings.polar):
        minbarvalue = 10
        if (not settings.bars):
            minbarvalue = 0
        func = np.vectorize(getPolarValue)
        polarizedspectrogram = func(spectrogram, maxbarvalue, minbarvalue)
        spectrogram = polarizedspectrogram

    if (settings.bars and settings.bar_scale != 1.0):
        scaledSpec = np.vectorize(applyBarScale)(spectrogram)
        spectrogram = scaledSpec

    return spectrogram, freqaxis, maxbarvalue

def arrange_freq_axis_for_chart_type(freqaxisData):
    if settings.polar:
        if settings.bars:
            indexes = list(range(1, len(freqaxisData) + 1))
            return [element * getBarWidth(len(freqaxisData)) for element in indexes]
        else:
            return np.linspace(0, 2. * np.pi, num=settings.number_of_points)
    return freqaxisData

def applyBarScale(value):
    return value * settings.bar_scale

def makeVideo(fig, freqaxis, dataLength):
    print('Creating animation...')
    # create video
    percentageTicks = getPercentageTicksArray(dataLength)
    anim = animation.FuncAnimation(fig, animate, frames=dataLength - 1, init_func=initAnimation,
                                   fargs=(freqaxis, percentageTicks, drawables), blit = True,
                                   interval=(1 / settings.framerate) * 1000)
    writervideo = animation.FFMpegWriter(fps=settings.framerate)
    animationfilename = pathlib.Path(settings.audio_file_path).stem + ' anim.mp4'
    animStartTime = datetime.datetime.now()
    anim.save(animationfilename, writer=writervideo, dpi=settings.dpiMultiplier)
    delta = datetime.datetime.now() - animStartTime
    print('Animating took ' + str(delta.seconds) + ' seconds.')
    print('Animation completed. Rendering and saving animation (this can take a while)...')
    input_video = ffmpeg.input(animationfilename)
    input_audio = ffmpeg.input('./' + pathlib.Path(settings.audio_file_path).stem + '.wav')
    filename = pathlib.Path(settings.audio_file_path).stem + datetime.datetime.now().strftime("%m%d-%H-%M-%S") + '.mp4'
    print('Saving...')
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename, **{'c:v': 'libx265'}, crf=20).run()
    print('Waveform complete!')
    delta2 = datetime.datetime.now() - animStartTime
    print('Overall, took about ' + str(delta2.seconds) + ' seconds to render a ' + str((dataLength-1)/settings.framerate) + ' second video.')

def getGhostDecay(visualizerData, rate_of_decay):
    decay_data = [np.array([0] * len(visualizerData[0])) for _ in range(len(visualizerData))]
    freq_decay_counter = [1] * len(visualizerData[0])
    framesBeforeDecrease = round(settings.framerate/6) #1/x of a second seems fine right?
    for i in range(len(visualizerData)):
        if i == 0:
            decay_data[i] = visualizerData[i]
            continue
        for freqIdx in range(len(visualizerData[i])):
            if (visualizerData[i][freqIdx] >= decay_data[i-1][freqIdx]):
                decay_data[i][freqIdx] = visualizerData[i][freqIdx]
                freq_decay_counter[freqIdx] = 1
            else:
                if (freq_decay_counter[freqIdx] > framesBeforeDecrease and decay_data[i-1][freqIdx] >= rate_of_decay):
                    decay_data[i][freqIdx]= decay_data[i-1][freqIdx] + rate_of_decay
                else:
                    decay_data[i][freqIdx] = decay_data[i - 1][freqIdx]
                freq_decay_counter[freqIdx] += 1


    return np.array(decay_data)

def createDrawables(fig, spectrumToShow, freqaxis, maxbarvalue):
    visAxisType = AxisType.CARTESIAN
    visShapeType = ShapeType.LINE
    if (settings.polar):
        visAxisType = AxisType.POLAR
    if (settings.bars):
        visShapeType = ShapeType.BAR

    #create the ghost drawable
    if (getAxis(fig, -10) != None):
        ghost_color = '#2F2F2F'
        ghostDecayRate = (-maxbarvalue) / settings.framerate
        ghost_decay_data = getGhostDecay(spectrumToShow, ghostDecayRate / 2)
        ghostArtistData = getArtistData(freqaxis, ghost_decay_data[0], getAxis(fig, -10), ghost_color, ghost_color)
        ghost_drawable = Drawable(ghostArtistData, ghost_decay_data, visAxisType, visShapeType, maxbarvalue)
        drawables.append(ghost_drawable)

    #create the main plot drawable
    artistData = getArtistData(freqaxis, spectrumToShow[0], getAxis(fig, 0), settings.bar_color, settings.bar_edge_color)
    visualizer_drawable = Drawable(artistData, spectrumToShow, visAxisType, visShapeType, maxbarvalue)
    drawables.append(visualizer_drawable)

    #create the background drawables, if necessary
    if (getAxis(fig, -99) != None):
        bkAxis = getAxis(fig, -99)
        bkArtists = []
        if (len(bkAxis.texts) > 0):
            bkArtists.append(bkAxis.texts)
        if (len(bkAxis.images) > 0):
            bkArtists.append(bkAxis.images)
        background_drawable = Drawable(bkArtists, None, AxisType.IMAGE, ShapeType.IMAGE)
        drawables.append(background_drawable)


def getArtistData(freqaxis, firstFrameData, axis, fillColor, edgeColor):
    artistData = []
    if (settings.bars):
        artistData = getBarData(freqaxis, firstFrameData, axis, fillColor, edgeColor)
    else:
        artistData = getLineData(freqaxis, axis, edgeColor, fillColor)
        fillArtist = axis.fill_between(freqaxis, 0, firstFrameData, facecolor = fillColor)
        artistData.append(fillArtist)
    return artistData


def getPercentageTicksArray(totalNumFrames):
    tenPercentTicks = []
    tenPercent = round(totalNumFrames / 10)
    for num in range(totalNumFrames):
        if (num != 0 and num % tenPercent == 0):
            tenPercentTicks.append(num)
    return tenPercentTicks

def getLineData(freqaxis, ax, linecolor, fillcolor):
    line = ax.plot(freqaxis, color=linecolor)
    return line

def getBarData(freqaxis, firstFrameData, ax, barcolor, baredgecolor):

    if (settings.polar):
        return ax.bar(freqaxis, firstFrameData, align='center', width=getBarWidth(len(freqaxis)), bottom=20, color=barcolor,
                      linewidth=1, edgecolor=baredgecolor)
    else:
        return ax.bar(freqaxis, firstFrameData, align='center', width=getBarWidth(), color=barcolor, linewidth=1, edgecolor=baredgecolor, zorder=ax.zorder)

def getBarWidth(axisLength = 0):
    # 75 width at 80 bars is perfect. These aren't scientific numbers, but they're a nice baseline.
    # TODO: make this look not-bad at other resolutions
    defaultWidth = 75
    defaultBars = 80
    barwidth = (defaultWidth) * (defaultBars / settings.number_of_points)
    if (settings.polar):
        barwidth = 2 * np.pi / axisLength
    return barwidth

def initAnimation():
    artists = []
    for item in drawables:
        if (item.shape_type == ShapeType.BAR):
            artists.extend(list(item.artist_info))
        if (item.shape_type == ShapeType.LINE):
            artists.extend(list(item.artist_info))
        if (item.axis_type == AxisType.IMAGE):
            for artist in item.artist_info:
                artists.extend(list(artist))

    return artists

def animate(frame, freqaxis, progress, drawables):

    artists = []
    for item in drawables:
        if (item.shape_type == ShapeType.BAR):
            draw_bars(item, frame)
            artists.extend(list(item.artist_info))
        if (item.shape_type == ShapeType.LINE):
            draw_line(item, freqaxis, frame)
            artists.extend(list(item.artist_info))

########################################
    if frame in progress:
        progress.pop(0)
        print('Animation ' + str(100 - (len(progress) * 10)) + '% complete')
    return artists

def draw_bars(drawable, frame_num):
    for i in range(len(drawable.data[frame_num])):
        drawable.artist_info[i].set_height(drawable.data[frame_num][i])

def draw_line(drawable, freqaxis, frame_num):

    drawable.artist_info[0].set_data(freqaxis, drawable.data[frame_num])

    drawable.artist_info[1].axes.collections.clear()
    fillcolor = settings.bar_color
    if (drawable.artist_info[1].axes.zorder == -10):
        fillcolor = '#2F2F2F'
    drawable.artist_info[1].axes.fill_between(freqaxis, 0, drawable.data[frame_num], facecolor = fillcolor)

def getPolarValue(value, maxValue, minValue = 10):
    slope = (maxValue - minValue) / (maxValue)
    return slope * (value) + minValue

def configurePlot():
    if (settings.polar):
        fig, visualizer_axis = pyplot.subplots(subplot_kw=dict(projection="polar"))
        cleanPolarPlot(visualizer_axis)
    else:
        fig, visualizer_axis = pyplot.subplots()
        cleanCartesianPlot(visualizer_axis)

    if (settings.ghost):
        addGhostVisualizer(fig)
    fig.set_size_inches(settings.resolution_x/settings.dpiMultiplier, settings.resolution_y/settings.dpiMultiplier)

    setAxis(visualizer_axis)


    createBackground(fig, visualizer_axis)
    if (settings.text != ''):
        addText(fig)

    return fig

def setAxis(axis):

    axis.set_ylim([0, 80])
    if (not settings.bars) and (not settings.polar):
        axis.set_xlim([0, settings.max_frequency])


def cleanPolarPlot(axis):
    pyplot.axis('off')
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)


def cleanCartesianPlot(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])


def createBackground(fig, ax):
    if (settings.bk_img_path != ''):
        print('!!! IMAGE SPECIFIED, THIS WILL TAKE A LOT LONGER TO RENDER, SORRY!!!')
        img = pyplot.imread(settings.bk_img_path)
        ax_image = createBackgroundAxisIfNotExists(fig)
        cleanCartesianPlot(ax_image)
        ax_image.imshow(img, extent=(0,settings.resolution_x,0,settings.resolution_y), resample=False)
        fig.patch.set_visible(False)
    elif (settings.bk_img_color != ''):
        ax_image = createBackgroundAxisIfNotExists(fig)
        ax_image.set_facecolor(settings.bk_img_color)
        fig.set_facecolor(settings.bk_img_color)
    else:
        fig.patch.set_visible(False)
        pyplot.rcParams.update({
            "figure.facecolor": (1.0, 0.0, 0.0, 0.0),  # red   with alpha = 0%
            "axes.facecolor": (0.0, 1.0, 0.0, 0.0),  # green with alpha = 0%
            "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),  # blue  with alpha = 0%
            "axes.ymargin": 0.,
            "axes.xmargin": 0.
        })

    ax.patch.set_visible(False)

def addText(fig):
    ax_image = createBackgroundAxisIfNotExists(fig)
    message = ax_image.text(0.5, 0.05, settings.text, fontsize='50', horizontalalignment='center', verticalalignment='center',
                            color=settings.text_color,
                            transform=ax_image.transAxes)
    if (settings.use_text_outline):
        message.set_path_effects([path_effects.Stroke(linewidth=settings.text_outline_width, foreground=settings.text_outline_color),
                                  path_effects.Normal()])

def addGhostVisualizer(fig):
    if (settings.polar):
        ghost_axis = fig.add_subplot(polar=True, zorder=-10)
        cleanPolarPlot(ghost_axis)
    else:
        ghost_axis = fig.add_subplot(zorder=-10)
        cleanCartesianPlot(ghost_axis)
    setAxis(ghost_axis)
    ghost_axis.patch.set_visible(False)

def createBackgroundAxisIfNotExists(fig):
    for axis in fig.axes:
        if axis.zorder == -99:
            return axis
    return fig.add_axes([0, 0, 1, 1], label="background", zorder=-99)

def getAxis(fig, zorder):
    return next((axis for axis in fig.axes if axis.zorder == zorder), None)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data
    with open('./config.json') as f:
        json_settings = json.load(f)

    settings = Settings(json_settings)
    drawables = []

    generate_visualizer()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
