import datetime
import pathlib
from datetime import time
import matplotlib.pyplot as pyplot
import matplotlib.animation as anim
import numpy as np
import ffmpeg
import librosa
import pydub
import json
import matplotlib.patheffects as path_effects

from drawable import DrawType, ShapeType, AxisType

pyplot.rcParams['animation.ffmpeg_path'] = '.\\ffmpeg.exe'
from matplotlib import animation


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
    spectrogram = spectrogram[0:len(freqaxis)]

    maxbarvalue = -79.9;
    if (number_of_bars != 0):
        newspec = []
        newfreq = []
        vals = np.arange(0, len(spectrogram) - int(len(spectrogram) / number_of_bars),
                         int(len(spectrogram) / number_of_bars))
        for i in vals:
            newspec.append(spectrogram[i])
            for j in spectrogram[i]:
                if (j > maxbarvalue):
                    maxbarvalue = j
            newfreq.append(freqaxis[i])

        spectrogram = np.array(newspec)
        freqaxis = np.array(newfreq)

    spectrogram = spectrogram.T
    return spectrogram, freqaxis, maxbarvalue


def makeVideo(spectrumToShow, freqaxis, maxbarvalue):
    print('Creating animation...')
    # create video

    #ax exists on fig.axes. maybe we need a better way to do this.
    #image object also needs to be blitted maybe for speed.
    fig, ax, line = configurePlot()
    axes = fig.axes
    #img = axes[1].images[0]
    barData = getBarData(freqaxis, spectrumToShow[0], ax)
    percentageTicks = getPercentageTicksArray(spectrumToShow)

    #don't pass in line, barData, img, whatever. Pass in a list of objects to display, then build out a list of artists to update.
    #drawabes might still not work
    drawables = []
    anim = animation.FuncAnimation(fig, animate, frames=len(spectrumToShow) - 1,
                                   fargs=(freqaxis, spectrumToShow, line, percentageTicks, barData, maxbarvalue, drawables),
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


def getPercentageTicksArray(dataToPlot):
    tenPercentTicks = []
    tenPercent = round(len(dataToPlot) / 10)
    for num in range(len(dataToPlot)):
        if (num != 0 and num % tenPercent == 0):
            tenPercentTicks.append(num)
    return tenPercentTicks


def getBarData(freqaxis, firstFrameData, ax):
    if (number_of_bars != 0):
        # 50 width at 80 bars is perfect. These aren't scientific numbers, but they're a nice baseline.
        defaultWidth = 75
        defaultBars = 80
        barwidth = (defaultWidth) * (defaultBars / number_of_bars)
        if (polar):
            barwidth = 2 * np.pi / len(freqaxis)
            indexes = list(range(1, len(freqaxis) + 1))
            freqaxis = [element * barwidth for element in indexes]
            return ax.bar(freqaxis, firstFrameData, align='center', width=barwidth, bottom=20, color=bar_color,
                          linewidth=1, edgecolor=bar_edge_color)
        else:
            return ax.bar(freqaxis, firstFrameData, align='center', width=barwidth, color=bar_color, linewidth=1, edgecolor=bar_edge_color)
    else:
        return -1

def animate(frame, freqaxis, stftArray, line, progress, barData, maxbarvalue, drawables):

######replacement for below#############
    for item in drawables:
        if (item.graph_type == ShapeType.BAR):
            draw_bars(item, frame)
        if (item.graph_type == ShapeType.LINE):
            draw_line(item, freqaxis)
########################################

    if number_of_bars != 0:
        dbAdjustment = 80

        for i in range(len(stftArray[frame])):
            if (polar):
                lowerlimit = 10
                slope = (maxbarvalue + dbAdjustment - lowerlimit) / (maxbarvalue + dbAdjustment)
                height = slope * (dbAdjustment + stftArray[frame][i]) + lowerlimit
                barData[i].set_height(height)
            else:
                barData[i].set_height((dbAdjustment + stftArray[frame][i]) * bar_scale)
    else:
        line.set_data(freqaxis, stftArray[frame])
    if frame in progress:
        progress.pop(0)
        print('Animation ' + str(100 - (len(progress) * 10)) + '% complete')
    return barData

def draw_bars(drawable, frame_num):
    dbAdjustment = 80

    for i in range(len(drawable.artist_info[frame_num])):
        if (drawable.axis_type == AxisType.POLAR):
            lowerlimit = 10
            slope = (drawable.max_value + dbAdjustment - lowerlimit) / (drawable.max_value + dbAdjustment)
            height = slope * (dbAdjustment + drawable.data[frame_num][i]) + lowerlimit
            drawable.artist_info[i].set_height(height)
        elif (drawable.axis_type == AxisType.CARTESIAN):
            drawable.artist_info[i].set_height((dbAdjustment + drawable.data[frame_num][i]) * bar_scale)

def draw_line(drawable, freqaxis, frame_num):
    drawable.artist_info[0].set_data(freqaxis, drawable.data[frame_num])


def configurePlot():
    if (polar):
        fig, ax = pyplot.subplots(subplot_kw=dict(projection="polar"))
        cleanPolarPlot(ax)

    else:
        fig, ax = pyplot.subplots()
        cleanCartesianPlot(ax)

    fig.set_size_inches(resolution_x/dpiMultiplier, resolution_y/dpiMultiplier)
    line, = ax.plot([], )


    if number_of_bars != 0:
        ax.set_ylim([0, 80])

    else:
        ax.set_xlim(0, max_frequency)
        ax.set_ylim([-80, 0])

    createBackground(fig, ax)

    return fig, ax, line


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

        ax_image = fig.add_axes([0, 0, 1, 1], label="imagegraph", zorder=-99)
        cleanCartesianPlot(ax_image)

        #ax_image.imshow(img, extent=(0,resolution_x,0,resolution_y), resample=False)

        if (text != ''):
            MESSAGE = ax_image.text(0.5, 0.1, text, fontsize='69', horizontalalignment='center', verticalalignment='center', color=text_color,
                          transform=ax_image.transAxes)
            if (use_text_outline):
                MESSAGE.set_path_effects([path_effects.Stroke(linewidth=text_outline_width, foreground=text_outline_color),
                                       path_effects.Normal()])
        ax.patch.set_visible(False)
        fig.patch.set_visible(False)
    elif (bk_img_color != ''):
        ax.set_facecolor(bk_img_color)
        fig.set_facecolor(bk_img_color)
    else:
        ax.patch.set_visible(False)
        fig.patch.set_visible(False)
        pyplot.rcParams.update({
            "figure.facecolor": (1.0, 0.0, 0.0, 0.0),  # red   with alpha = 0%
            "axes.facecolor": (0.0, 1.0, 0.0, 0.0),  # green with alpha = 0%
            "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),  # blue  with alpha = 0%
            "axes.ymargin": 0.,
            "axes.xmargin": 0.
        })

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

    number_of_bars = data["number_of_bars"]
    polar = data["polar"]

    # I'm pretty sure i can get a neat polar line. Might redo this later.
    if (number_of_bars == 0):
        polar = False
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
    dpiMultiplier = 100
    parse_audio()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
