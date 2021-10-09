from datetime import time

import soundfile as sf
import matplotlib.pyplot as pyplot
import matplotlib.animation as anim
import scipy.fftpack as fftpack
import numpy as np

monodata = []

def parse_audio_filepath(path):
    return sf.SoundFile(path)


def parse_audio(audioData):
    audioArray = audioData.read()
    for dat in audioArray:
        monodata.append((dat[0] + dat[1])/2)

    #plot_audioTimeData(monodata)


    #pyplot.ion()
    #pyplot.show()

    plot_audioFreqData(monodata, int(audioData.samplerate), 1/30, 0.025)



def plot_audioTimeData(audioData):
    pyplot.subplot(211)
    pyplot.plot(getTimeAxis(audioData), audioData)
    pyplot.title('False Knight, 10-20s, mono')


def plot_audioFreqData(audioData, samplerate, skipParamMultiplier, windowMultiplier):
    max_freq = 20000
    tentwentyfour = 1024
    updateInterval = samplerate / 30  # 30fps
    skipParam = int(samplerate * skipParamMultiplier)
    window = int(samplerate * windowMultiplier)
    frames = enframe(audioData, skipParam, window)
    (shortTimeFourierTransform, freqaxis) = stft(frames, len(frames), samplerate)

    #pyplot.subplot(211)

    fig = pyplot.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlim(0, 20000)
    #ax.set_ylim([-10, 2])
    ax.set_ylim([-0, 7.5])
    line, = ax.plot([], )
    fig.canvas.draw()
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
    pyplot.show(block= False)
    for i in range(len(frames)):
        #if int(i%updateInterval) == 0:
        print(list[i])
        #line.set_data(freqaxis[freqaxis <= max_freq], np.log(abs(shortTimeFourierTransform[i][freqaxis <= max_freq])))
        line.set_data(freqaxis[freqaxis <= 20000], np.log(np.maximum(1, abs(shortTimeFourierTransform[i][freqaxis <= 20000]) ** 2)))
        fig.canvas.restore_region(axbackground)
        ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)
        #drawGraphs(shortTimeFourierTransform, freqaxis, max_freq, i)
        pyplot.pause(1 / 30)
            #framesToDraw.append(audioData[i])
            #time.sleep(1/updateInterval)



    # 5. Save the animation
    # anim.save(
    #    filename='/tmp/sn2011fe_spectral_time_series.mp4',
    #    fps=24,
    #    extra_args=['-vcodec', 'libx264'],
    #    dpi=300,
    #)

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
    # w = 0.54*np.ones(L)
    # for n in range(0,L):
    #   w[n] = w[n] - 0.46*math.cos(2*math.pi*n/(L-1))
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
    audioFile = parse_audio_filepath('fk.wav')
    parse_audio(audioFile)

    input('Press Enter to exit')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
