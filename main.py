import datetime
from datetime import time

import soundfile as sf
import matplotlib.pyplot as pyplot
import matplotlib.animation as anim
import scipy.fftpack as fftpack
import numpy as np
import ffmpeg
#pyplot.rcParams['animation.ffmpeg_path'] = 'X:\\SW Libraries\\ffmpeg\\bin'
from matplotlib import animation

monodata = []

def parse_audio_filepath(path):
    return sf.SoundFile(path)


def parse_audio(audioData, maxfreq, framerate, blankSecondsAfter, bars):
    audioArray = audioData.read()
    for dat in audioArray:
        monodata.append((dat[0] + dat[1])/2)

    blankFrames = blankSecondsAfter * framerate
    for blankFrame in range(blankFrames):
        monodata.append(0)

    #plot_audioTimeData(monodata)


    #pyplot.ion()
    #pyplot.show()

    plot_audioFreqData(monodata, int(audioData.samplerate), 1/framerate, 0.025, maxfreq, bars)



def plot_audioTimeData(audioData):
    pyplot.subplot(211)
    pyplot.plot(getTimeAxis(audioData), audioData)
    pyplot.title('False Knight, 10-20s, mono')


def plot_audioFreqData(audioData, samplerate, skipParamMultiplier, windowMultiplier, maxfreq, bars):
    fig, ax = pyplot.subplots()
    fig.set_size_inches(19.2, 10.8)
    configurePlot(fig, ax, maxfreq, bars)
    line, = ax.plot([], )
    updateInterval = samplerate / framerate
    #skip parameter multiplier is one over the framerate. we can use it as a multiplier to pass less parameters around.
    skipParam = int(samplerate * skipParamMultiplier)
    #window is frame length. decreasing it increases smear which could look more pleasing.
    window = int(samplerate * windowMultiplier)
    print('Generating spectrogram array...')
    frames = enframe(audioData, skipParam, window)
    (shortTimeFourierTransform, freqaxis) = stft(frames, int(len(frames)/80) if bars else len(frames), samplerate)

    #spectrumToShow = stft2level(shortTimeFourierTransform, round(len(frames) * maxfreq / samplerate)) if bars else shortTimeFourierTransform
    spectrumToShow = stft2level(shortTimeFourierTransform,
                                len(frames)) if bars else shortTimeFourierTransform
    interval = int(len(spectrumToShow[0]) / 80)
    newSpecToShow = []

    print('Creating animation...')
    #create video
    tenPercentTicks = []
    tenPercent = round(len(spectrumToShow)/10)
    for num in range(len(spectrumToShow)):
        if (num != 0 and num%tenPercent == 0):
            tenPercentTicks.append(num)
    anim = animation.FuncAnimation(fig, animate, frames=len(frames)-1, fargs=(freqaxis, spectrumToShow, line, maxfreq, tenPercentTicks, bars),
                                   interval=(skipParamMultiplier)*1000)

    fpsCalc = int(1/skipParamMultiplier)
    writervideo = animation.FFMpegWriter(fps=fpsCalc)
    anim.save('./func.mp4', writer=writervideo, dpi=100)
    print('Animation completed. Rendering and saving animation (this can take a while)...')
    input_video = ffmpeg.input('./func.mp4').filter('scale', 1080, -1)
    input_audio = ffmpeg.input('./fk.wav')
    filename = './song' + datetime.datetime.now().strftime("%H-%M-%S") + '.mp4'
    print('Saving...')
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename, **{'c:v': 'libx265'}, crf=20).run()
    print('Waveform complete!')

def animate(frame, axis, stftArray, line, maxfreq, progress, bars):
    parsedAxis = axis[axis <= maxfreq]
    line.set_data(axis[axis <= maxfreq],stftArray[frame][axis <= maxfreq]) if bars else line.set_data(axis[axis <= maxfreq], np.log(np.maximum(1, abs(stftArray[frame][axis <= maxfreq]) ** 2)))
    if frame in progress:
        progress.pop(0)
        print('Animation ' + str(100 - (len(progress)*10)) + '% complete')
    #np.log(np.maximum(1, abs(stftArray[frame][axis <= maxfreq]) ** 2)))
    #line.set_data(axis, stftArray[frame])
    return line

def configurePlot(fig, ax, maxfreq, bars) :
    ax.set_xlim(0, maxfreq)
    ax.set_ylim([-10 if bars else -2, 60 if bars else 10])
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

def drawGraphs(stft, freqaxis, max_freq, frame):
    #pyplot.clf()
    #pyplot.plot(211)
    #pyplot.ylabel('Magnitude Squared STFT')
    #pyplot.ylim(-10, 2)
    #pyplot.xlabel('Frequency (Hertz)')
    #pyplot.title('Spectrum of a frame from False Knight')


    pyplot.plot(freqaxis[freqaxis <= max_freq], np.log(abs(stft[frame][freqaxis <= max_freq])))


    # spectrogram = stft2level(shortTimeFourierTransform, int(tentwentyfour * max_freq / samplerate))
    # pyplot.plot(spectrogram)

    # this looks good for a line spectrogram
    #pyplot.plot(212)
    #pyplot.plot(freqaxis[freqaxis <= 20000], np.log(np.maximum(1, abs(stft[frame][freqaxis <= 20000]) ** 2)))

    #pyplot.draw()

    # pyplot.plot(getTimeAxisAtFrame(frames, 300), frames[300])
    #pyplot.draw()

    #fig.canvas.restore_region(background)

    # redraw just the points
    #ax.draw_artist(freqaxis[freqaxis <= max_freq], np.log(abs(stft[frame][freqaxis <= max_freq])))

    # fill in the axes rectangle
    #fig.canvas.blit(ax.bbox)

def getTimeAxis(audioData):
    return np.linspace(10, 20, len(audioData))


def getTimeAxisAtFrame(frames, frameNum):
    return np.linspace(0, 0.005, len(frames[frameNum]))


def enframe(signal, skipParameter, windowLength):
    w = np.hamming(windowLength)
    frames = []
    nframes = 1 + int((len(signal) - windowLength) / skipParameter)
    for t in range(0, nframes):
        frames.append(np.copy(signal[(t * skipParameter):(t * skipParameter + windowLength)]) * w)
    return (frames)

def stft(frames,fftLength,samplingFreq):
    stft_frames = [ fftpack.fft(x,fftLength) for x in frames]
    freq_axis = np.linspace(0,samplingFreq,fftLength)
    return(stft_frames, freq_axis)

def stft2level(stft_spectra,max_freq_bin):
    magnitude_spectra = [ abs(x) for x in stft_spectra ]
    max_magnitude = max([ max(x) for x in magnitude_spectra ])
    min_magnitude = max_magnitude / 1000.0
    for t in range(0,len(magnitude_spectra)):
        for k in range(0,len(magnitude_spectra[t])):
            magnitude_spectra[t][k] /= min_magnitude
            if magnitude_spectra[t][k] < 1:
                magnitude_spectra[t][k] = 1
    level_spectra = [ 20*np.log10(x[0:max_freq_bin]) for x in magnitude_spectra ]
    return(level_spectra)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data

    bars = False
    maxfreq = 8000
    framerate = 30
    blankSecondsAfter = 2
    audioFile = parse_audio_filepath('fk.wav')
    parse_audio(audioFile, maxfreq, framerate, blankSecondsAfter, bars)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
