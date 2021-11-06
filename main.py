import datetime
from datetime import time
import json
import pathlib
from pathlib import Path
import logging
import ffmpeg
import librosa
from matplotlib import animation
import matplotlib.animation as anim
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as pyplot
import numpy as np
import pydub

import drawable
from drawable import AxisType, ShapeType, Drawable

def determineFfmpegPath():
    path = Path('.\\ffmpeg.exe')

    if (path.is_file()):
        return '.\\ffmpeg.exe'

    return 'ffmpeg'

pyplot.rcParams['animation.ffmpeg_path'] = determineFfmpegPath()

def parse_audio():
    plot_audioFreqDataLibrosa()


def loadFromFile():
    filepath = audio_file_path
    filetype = pathlib.Path(filepath).suffix
    filename = pathlib.Path(filepath).stem
    print('File type is ' + filetype)
    if (filetype == '.mp3'):
        print('Converting to .wav...')
        audiodata = pydub.AudioSegment.from_mp3(audio_file_path)
        newFilePath = './' + filename + '.wav'
        print('Saving new .wav file...')
        audiodata.export(newFilePath, format='wav')
        print('File saved.')
        filepath = newFilePath

    print('Loading data from audio file.')
    return librosa.load(filepath)


def plot_audioFreqDataLibrosa():
    timeseries, samplerate = loadFromFile()
    spectrogram, freqaxis, maxbarvalue = getSpectrogramAndFrequencyAxisAndMaxDb(timeseries, samplerate)

    makeVideo(spectrogram, freqaxis, maxbarvalue)


def getSpectrogramAndFrequencyAxisAndMaxDb(timeseries, samplerate):
    # skip parameter multiplier is one over the framerate. we can use it as a multiplier to pass less parameters around.
    skipParam = int(samplerate * (1 / framerate))

    shortTimeFourierTransform = np.abs(librosa.stft(timeseries, hop_length=skipParam, n_fft=2048 * 4))
    spectrogram = librosa.amplitude_to_db(shortTimeFourierTransform, ref=np.max)

    freqaxis = librosa.core.fft_frequencies(samplerate, 2048 * 4)

    freqaxisTemp = [x for x in freqaxis if x <= max_frequency]

    freqaxis = freqaxisTemp
    spectrogram = spectrogram[0:len(freqaxis)] + 80

    maxbarvalue = 0.1;

    newspec = []
    newfreq = []
    vals = np.arange(0, len(spectrogram) - int(len(spectrogram) / number_of_points),
                     int(len(spectrogram) / number_of_points))
    for i in vals:
        newspec.append(np.array(spectrogram[i]))
        for j in spectrogram[i]:
            if (j > maxbarvalue):
                maxbarvalue = j
        newfreq.append(freqaxis[i])

    freqaxis = np.array(arrange_freq_axis_for_chart_type(newfreq))
    spectrogram = np.array(newspec)

    spectrogram = spectrogram.T
    if (polar):
        minbarvalue = 10
        if (not bars):
            minbarvalue = 0
        func = np.vectorize(getPolarValue)
        polarizedspectrogram = func(spectrogram, maxbarvalue, minbarvalue)
        spectrogram = polarizedspectrogram

    if (bars and bar_scale != 1.0):
        scaledSpec = np.vectorize(applyBarScale)(spectrogram)
        spectrogram = scaledSpec

    return spectrogram, freqaxis, maxbarvalue

def arrange_freq_axis_for_chart_type(freqaxisData):
    if polar:
        if bars:
            indexes = list(range(1, len(freqaxisData) + 1))
            return [element * getBarWidth(len(freqaxisData)) for element in indexes]
        else:
            return np.linspace(0, 2. * np.pi, num=number_of_points)
    return freqaxisData

def applyBarScale(value):
    return value * bar_scale

def makeVideo(spectrumToShow, freqaxis, maxbarvalue):
    print('Creating animation...')
    # create video

    #ax exists on fig.axes. maybe we need a better way to do this.
    #image object also needs to be blitted maybe for speed.
    fig = configurePlot()

    visAxisType = AxisType.CARTESIAN
    visShapeType = ShapeType.LINE
    if (polar):
        visAxisType = AxisType.POLAR


    if (bars):
        visShapeType = ShapeType.BAR

    if (getAxis(fig, -10) != None):
        ghost_color = '#2F2F2F'
        blank_data = []
        ghost_frames = int(framerate/4)
        ghostDecayRate = (-maxbarvalue)/framerate
        ghost_decay_data = getGhostDecay(spectrumToShow, ghostDecayRate/2)

        ghostArtistData = []
        if (bars):
            ghostArtistData = getBarData(freqaxis, ghost_decay_data[0], getAxis(fig, -10), ghost_color, ghost_color)
        else:
            ghostArtistData = getLineData(freqaxis, getAxis(fig, -10), ghost_color, ghost_color)
            ghostFillArtist = getAxis(fig, -10).fill_between(freqaxis, 0, ghost_decay_data[0], facecolor = ghost_color)
            ghostArtistData.append(ghostFillArtist)
        ghost_drawable = Drawable(ghostArtistData, ghost_decay_data, visAxisType, visShapeType, maxbarvalue)
        drawables.append(ghost_drawable)


    artistData = []
    if (bars):
        artistData = getBarData(freqaxis, spectrumToShow[0], getAxis(fig, 0), bar_color, bar_edge_color)
    else:
        artistData = getLineData(freqaxis, getAxis(fig, 0), bar_edge_color, bar_color)
        fillArtist = getAxis(fig,0).fill_between(freqaxis, 0, spectrumToShow[0], facecolor = bar_color)
        artistData.append(fillArtist)
    visualizer_drawable = Drawable(artistData, spectrumToShow, visAxisType, visShapeType, maxbarvalue)
    drawables.append(visualizer_drawable)

    if (getAxis(fig, -99) != None):
        bkAxis = getAxis(fig, -99)
        bkArtists = []
        if (len(bkAxis.texts) > 0):
            bkArtists.append(bkAxis.texts)
        if (len(bkAxis.images) > 0):
            bkArtists.append(bkAxis.images)
        background_drawable = Drawable(bkArtists, None, AxisType.IMAGE, ShapeType.IMAGE)
        drawables.append(background_drawable)

    percentageTicks = getPercentageTicksArray(len(spectrumToShow))

    anim = animation.FuncAnimation(fig, animate, frames=len(spectrumToShow) - 1, init_func=initAnimation,
                                   fargs=(freqaxis, percentageTicks, drawables), blit = True,
                                   interval=(1 / framerate) * 1000)

    writervideo = animation.FFMpegWriter(fps=framerate)
    animationfilename = pathlib.Path(audio_file_path).stem + ' anim.mp4'
    animStartTime = datetime.datetime.now()
    anim.save(animationfilename, writer=writervideo, dpi=dpiMultiplier)
    delta = datetime.datetime.now() - animStartTime
    print('Animating took ' + str(delta.seconds) + ' seconds.')
    print('Animation completed. Rendering and saving animation (this can take a while)...')
    input_video = ffmpeg.input(animationfilename)
    input_audio = ffmpeg.input('./' + pathlib.Path(audio_file_path).stem + '.wav')
    filename = pathlib.Path(audio_file_path).stem + datetime.datetime.now().strftime("%m%d-%H-%M-%S") + '.mp4'
    print('Saving...')
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename, **{'c:v': 'libx265'}, crf=20).run()
    print('Waveform complete!')
    delta2 = datetime.datetime.now() - animStartTime
    print('Overall, took about ' + str(delta2.seconds) + ' seconds to render a ' + str((len(spectrumToShow)-1)/framerate) + ' second video.')

def getGhostDecay(visualizerData, rate_of_decay):
    decay_data = [np.array([0] * len(visualizerData[0])) for _ in range(len(visualizerData))]
    freq_decay_counter = [1] * len(visualizerData[0])
    framesBeforeDecrease = round(framerate/6) #1/x of a second seems fine right?
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

    if (polar):
        return ax.bar(freqaxis, firstFrameData, align='center', width=getBarWidth(len(freqaxis)), bottom=20, color=barcolor,
                      linewidth=1, edgecolor=baredgecolor)
    else:
        return ax.bar(freqaxis, firstFrameData, align='center', width=getBarWidth(), color=barcolor, linewidth=1, edgecolor=baredgecolor, zorder=ax.zorder)

def getBarWidth(axisLength = 0):
    # 75 width at 80 bars is perfect. These aren't scientific numbers, but they're a nice baseline.
    # TODO: make this look not-bad at other resolutions
    defaultWidth = 75
    defaultBars = 80
    barwidth = (defaultWidth) * (defaultBars / number_of_points)
    if (polar):
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
    fillcolor = bar_color
    if (drawable.artist_info[1].axes.zorder == -10):
        fillcolor = '#2F2F2F'
    drawable.artist_info[1].axes.fill_between(freqaxis, 0, drawable.data[frame_num], facecolor = fillcolor)

def getPolarValue(value, maxValue, minValue = 10):
    slope = (maxValue - minValue) / (maxValue)
    return slope * (value) + minValue

def configurePlot():
    if (polar):
        fig, visualizer_axis = pyplot.subplots(subplot_kw=dict(projection="polar"))
        cleanPolarPlot(visualizer_axis)
    else:
        fig, visualizer_axis = pyplot.subplots()
        cleanCartesianPlot(visualizer_axis)

    if (ghost):
        addGhostVisualizer(fig)
    fig.set_size_inches(resolution_x/dpiMultiplier, resolution_y/dpiMultiplier)

    setAxis(visualizer_axis)


    createBackground(fig, visualizer_axis)
    if (text != ''):
        addText(fig)

    return fig

def setAxis(axis):

    axis.set_ylim([0, 80])
    if (not bars) and (not polar):
        axis.set_xlim([0, max_frequency])


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
    if (bk_img_path != ''):
        print('!!! IMAGE SPECIFIED, THIS WILL TAKE A LOT LONGER TO RENDER, SORRY!!!')
        img = pyplot.imread(bk_img_path)
        ax_image = createBackgroundAxisIfNotExists(fig)
        cleanCartesianPlot(ax_image)
        ax_image.imshow(img, extent=(0,resolution_x,0,resolution_y), resample=False)
        fig.patch.set_visible(False)
    elif (bk_img_color != ''):
        ax_image = createBackgroundAxisIfNotExists(fig)
        ax_image.set_facecolor(bk_img_color)
        fig.set_facecolor(bk_img_color)
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
    message = ax_image.text(0.5, 0.05, text, fontsize='50', horizontalalignment='center', verticalalignment='center',
                            color=text_color,
                            transform=ax_image.transAxes)
    if (use_text_outline):
        message.set_path_effects([path_effects.Stroke(linewidth=text_outline_width, foreground=text_outline_color),
                                  path_effects.Normal()])

def addGhostVisualizer(fig):
    if (polar):
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

def getxyResolutions(resolutionOption):
    if (resolutionOption == "4k"):
        return 3840, 2160
    elif (resolutionOption == "1080p"):
        return 1920, 1080
    elif (resolutionOption == "twitter"):
        return 1000, 1000
    elif (resolutionOption == "720p"):
        return 1280, 720
    else:
        return 1920, 1080
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data
    with open('./config.json') as f:
        data = json.load(f)

    number_of_points = data["number_of_bars"]
    polar = data["polar"]
    bars = True
    # I'm pretty sure i can get a neat polar line. Might redo this later.
    if (number_of_points == 0):
        bars = False
        number_of_points = 80
    #We need to do hacky shit to have a default amount of 80 line points.. ehhghhhhh


    # 10k is pretty good for straight bars, 8k looks good in a circle
    max_frequency = data["max_frequency"]
    bar_color = data["bar_color"]
    bar_edge_color = data["bar_edge_color"]
    if (bar_edge_color == ""):
        bar_edge_color = bar_color
    bar_scale = data["bar_scale"]
    framerate = data["framerate"]
    bk_img_path = data["background_img_path"]
    bk_img_color = data["background_color"]
    audio_file_path = data["audio_file_path"]
    resolution_type = data["resolution"]
    resolution_x, resolution_y = getxyResolutions(resolution_type)

    text = data["text"]
    text_color = data["text_color"]
    if (text_color == ''):
        text_color = '#FFFFFF'
    use_text_outline = data["use_text_outline"]
    text_outline_color = data["text_outline_color"]
    if (text_outline_color == ''):
        text_outline_color = text_color
    text_outline_width = data["text_outline_width"]
    ghost = data["ghost"]
    drawables = []
    dpiMultiplier = 100
    parse_audio()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
