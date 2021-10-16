import datetime
from datetime import time

import soundfile as sf
import matplotlib.pyplot as pyplot
import matplotlib.animation as anim
import scipy.fftpack as fftpack
import numpy as np
import ffmpeg
import librosa
pyplot.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
from matplotlib import animation

monodata = []


def parse_audio_filepath(path):
    return sf.SoundFile(path)


def parse_audio(audioData, maxfreq, framerate, bars, polar):
    audioArray = audioData.read()
    for dat in audioArray:
        monodata.append((dat[0] + dat[1]) / 2)

    plot_audioFreqDataLibrosa('fk.wav', 1/framerate, 0.025, bars, polar, maxfreq)


def plot_audioFreqDataLibrosa(audioFilename, skipParamMultiplier, windowMultiplier, bars, polar, maxfreq):
    timeseries, samplerate = librosa.load(audioFilename)
    # skip parameter multiplier is one over the framerate. we can use it as a multiplier to pass less parameters around.
    skipParam = int(samplerate * skipParamMultiplier)
    # window is frame length. decreasing it increases smear which could look more pleasing.
    window = int(samplerate * windowMultiplier)
    shortTimeFourierTransform = np.abs(librosa.stft(timeseries, hop_length=skipParam, n_fft=2048*4))
    spectrogram = librosa.amplitude_to_db(shortTimeFourierTransform, ref=np.max)

    freqaxis = librosa.core.fft_frequencies(samplerate, 2048*4)

    freqaxisTemp = [x for x in freqaxis if x <= maxfreq]

    freqaxis = freqaxisTemp
    spectrogram = spectrogram[0:len(freqaxis)]

    maxbarvalue = -79.9;
    if (bars != 0):
        newspec = []
        newfreq = []
        vals = np.arange(0, len(spectrogram) - int(len(spectrogram)/bars), int(len(spectrogram)/bars))
        for i in vals:
            newspec.append(spectrogram[i])
            for j in spectrogram[i]:
                if (j > maxbarvalue):
                    maxbarvalue = j
            newfreq.append(freqaxis[i])
        #run an average

        spectrogram = np.array(newspec)
        freqaxis = np.array(newfreq)

    spectrogram = spectrogram.T

    makeVideo(spectrogram, freqaxis, skipParamMultiplier, bars, polar, maxbarvalue)



def makeVideo(spectrumToShow, freqaxis, skipParamMultiplier, bars, polar, maxbarvalue):
    print('Creating animation...')
    # create video
    tenPercentTicks = []
    tenPercent = round(len(spectrumToShow) / 10)
    for num in range(len(spectrumToShow)):
        if (num != 0 and num % tenPercent == 0):
            tenPercentTicks.append(num)

    if (polar):
        fig, ax = pyplot.subplots(subplot_kw=dict(projection="polar"))
    else:
        fig, ax = pyplot.subplots()
    fig.set_size_inches(19.2, 10.8)
    ax.plot()
    configurePlot(fig, ax, maxfreq, bars, polar)
    line, = ax.plot([], )
    barData = -1


    if (bars != 0):
        #100 width at 80 bars is perfect. These aren't scientific numbers, but they're a nice baseline.
        defaultWidth = 100
        defaultBars = 80
        barwidth = (defaultWidth) * (defaultBars / bars)
        if (polar):
            barwidth = 2 * np.pi / len(freqaxis)
            indexes = list(range(1, len(freqaxis)+1))
            freqaxis = [element * barwidth for element in indexes]
            barData = ax.bar(freqaxis, spectrumToShow[0], align='center', width=barwidth, bottom = 20, color='black')
        else:
            barData = ax.bar(freqaxis, spectrumToShow[0], align='center', width=barwidth)



    anim = animation.FuncAnimation(fig, animate, frames=len(spectrumToShow) - 1,
                                   fargs=(freqaxis, spectrumToShow, line, maxfreq, tenPercentTicks, bars, barData, polar, maxbarvalue),
                                   interval=(skipParamMultiplier) * 1000)

    fpsCalc = int(1 / skipParamMultiplier)
    writervideo = animation.FFMpegWriter(fps=fpsCalc)
    anim.save('./func.mp4', writer=writervideo, dpi=100)
    print('Animation completed. Rendering and saving animation (this can take a while)...')
    input_video = ffmpeg.input('./func.mp4').filter('scale', 1080, -1)
    input_audio = ffmpeg.input('./fk.wav')
    filename = './song' + datetime.datetime.now().strftime("%H-%M-%S") + '.mp4'
    print('Saving...')
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename, **{'c:v': 'libx265'}, crf=20).run()
    print('Waveform complete!')

def animate(frame, axis, stftArray, line, maxfreq, progress, bars, barData, polar, maxbarvalue):
    if bars != 0:
        dbAdjustment = 80

        for i in range(len(stftArray[frame])):
            if (polar):
                lowerlimit = 10
                slope = (maxbarvalue + dbAdjustment - lowerlimit) / (maxbarvalue+dbAdjustment)
                height = slope * (dbAdjustment+ stftArray[frame][i]) + lowerlimit
                barData[i].set_height(height)
            else:
                barData[i].set_height(dbAdjustment+stftArray[frame][i])
    else:
        line.set_data(axis, stftArray[frame])#np.log(np.maximum(1, abs(stftArray[frame][axis <= maxfreq]) ** 2)))
    if frame in progress:
        progress.pop(0)
        print('Animation ' + str(100 - (len(progress) * 10)) + '% complete')
    return line


def configurePlot(fig, ax, maxfreq, bars, polar):
    if bars != 0:
        ax.set_ylim([0, 80])

    else:
        ax.set_xlim(0, maxfreq)
        ax.set_ylim([-80, 0])

    if (polar):
        pyplot.axis('off')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.patch.set_visible(False)
        fig.patch.set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.patch.set_visible(False)
        fig.patch.set_visible(False)
    pyplot.rcParams.update({
        "figure.facecolor": (1.0, 0.0, 0.0, 0.0),  # red   with alpha = 0%
        "axes.facecolor": (0.0, 1.0, 0.0, 0.0),  # green with alpha = 0%
        "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),  # blue  with alpha = 0%
    })

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data
    bars = 80
    polar = True
    if (bars == 0):
        polar = False
    maxfreq = 8000
    framerate = 30
    audioFile = parse_audio_filepath('fk.wav')
    parse_audio(audioFile, maxfreq, framerate, bars, polar)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
