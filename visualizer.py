import datetime
import pathlib
import sys
from pathlib import Path
import ffmpeg
import librosa
from matplotlib import animation
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as pyplot
import numpy as np
import pydub
from drawable import AxisType, ShapeType, Drawable
from settings import Settings
import logging

class Visualizer:

    def __init__(self, settings: Settings):
        self.settings = settings
        if (self.settings.number_of_points == 0):
            self.settings.bars = False
            self.settings.number_of_points = 80
        self.drawables = []
        pyplot.rcParams['animation.ffmpeg_path'] = self.determineFfmpegPath()
        logging.basicConfig(format='%(asctime)s %(message)s', filename='visualizer.log', encoding='utf-8', level=logging.INFO)
        sys.excepthook = self.unhandledExceptionCatcher

    def unhandledExceptionCatcher(self, exctype, value, tb):
        logging.info('ERROR')


    def determineFfmpegPath(self):
        path = Path('.\\ffmpeg.exe')

        if (path.is_file()):
            return '.\\ffmpeg.exe'

        return 'ffmpeg'



    def loadFromFile(self):
        filepath = self.settings.audio_file_path
        filetype = pathlib.Path(filepath).suffix
        filename = pathlib.Path(filepath).stem
        logging.info('File type is ' + filetype)
        if (filetype == '.mp3'):
            logging.info('Converting to .wav...')
            audiodata = pydub.AudioSegment.from_mp3(self.settings.audio_file_path)
            newFilePath = './' + filename + '.wav'
            logging.info('Saving new .wav file...')
            audiodata.export(newFilePath, format='wav')
            logging.info('File saved.')
            filepath = newFilePath

        logging.info('Loading data from audio file.')
        return librosa.load(filepath)

    def setSettings(self, newsettings):
        self.settings = newsettings

    def generate_visualizer(self):
        timeseries, samplerate = self.loadFromFile()
        spectrogram, freqaxis, maxbarvalue = self.getSpectrogramAndFrequencyAxisAndMaxDb(timeseries, samplerate)
        fig = self.configurePlot()
        self.createDrawables(fig, spectrogram, freqaxis, maxbarvalue)
        self.makeVideo(fig, freqaxis, len(spectrogram))

    def getSpectrogramAndFrequencyAxisAndMaxDb(self, timeseries, samplerate):
        # skip parameter multiplier is one over the framerate. we can use it as a multiplier to pass less parameters around.
        skipParam = int(samplerate * (1 / self.settings.framerate))

        shortTimeFourierTransform = np.abs(librosa.stft(timeseries, hop_length=skipParam, n_fft=2048 * 4))
        spectrogram = librosa.amplitude_to_db(shortTimeFourierTransform, ref=np.max)

        freqaxis = librosa.core.fft_frequencies(samplerate, 2048 * 4)

        freqaxisTemp = [x for x in freqaxis if x <= self.settings.max_frequency]

        freqaxis = freqaxisTemp
        spectrogram = spectrogram[0:len(freqaxis)] + 80

        maxbarvalue = 0.1;

        newspec = []
        newfreq = []
        vals = np.arange(0, len(spectrogram) - int(len(spectrogram) / self.settings.number_of_points),
                         int(len(spectrogram) / self.settings.number_of_points))
        for i in vals:
            newspec.append(np.array(spectrogram[i]))
            for j in spectrogram[i]:
                if (j > maxbarvalue):
                    maxbarvalue = j
            newfreq.append(freqaxis[i])

        freqaxis = np.array(self.arrange_freq_axis_for_chart_type(newfreq))
        spectrogram = np.array(newspec)

        spectrogram = spectrogram.T
        if (self.settings.polar):
            minbarvalue = 10
            if (not self.settings.bars):
                minbarvalue = 0
            func = np.vectorize(self.getPolarValue)
            polarizedspectrogram = func(spectrogram, maxbarvalue, minbarvalue)
            spectrogram = polarizedspectrogram

        if (self.settings.bars and self.settings.bar_scale != 1.0):
            scaledSpec = np.vectorize(self.applyBarScale)(spectrogram)
            spectrogram = scaledSpec

        return spectrogram, freqaxis, maxbarvalue

    def arrange_freq_axis_for_chart_type(self, freqaxisData):
        if self.settings.polar:
            if self.settings.bars:
                indexes = list(range(1, len(freqaxisData) + 1))
                return [element * self.getBarWidth(len(freqaxisData)) for element in indexes]
            else:
                return np.linspace(0, 2. * np.pi, num=self.settings.number_of_points)
        return freqaxisData

    def applyBarScale(self, value):
        return value * self.settings.bar_scale

    def makeVideo(self, fig, freqaxis, dataLength):
        logging.info('Creating animation...')
        # create video
        percentageTicks = self.getPercentageTicksArray(dataLength)
        anim = animation.FuncAnimation(fig, self.animate, frames=dataLength - 1, init_func=self.initAnimation,
                                       fargs=(freqaxis, percentageTicks, self.drawables), blit=True,
                                       interval=(1 / self.settings.framerate) * 1000)
        writervideo = animation.FFMpegWriter(fps=self.settings.framerate)
        animationfilename = pathlib.Path(self.settings.audio_file_path).stem + ' anim.mp4'
        animStartTime = datetime.datetime.now()
        anim.save(animationfilename, writer=writervideo, dpi=self.settings.dpiMultiplier)
        delta = datetime.datetime.now() - animStartTime
        logging.info('Animating took ' + str(delta.seconds) + ' seconds.')
        logging.info('Animation completed. Rendering and saving animation (this can take a while)...')
        input_video = ffmpeg.input(animationfilename)
        input_audio = ffmpeg.input('./' + pathlib.Path(self.settings.audio_file_path).stem + '.wav')
        filename = pathlib.Path(self.settings.audio_file_path).stem + datetime.datetime.now().strftime(
            "%m%d-%H-%M-%S") + '.mp4'
        logging.info('Saving...')
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename, **{'c:v': 'libx265'}, crf=20).run()
        logging.info('Waveform complete!')
        delta2 = datetime.datetime.now() - animStartTime
        logging.info('Overall, took about ' + str(delta2.seconds) + ' seconds to render a ' + str(
            (dataLength - 1) / self.settings.framerate) + ' second video.')

    def getGhostDecay(self, visualizerData, rate_of_decay):
        decay_data = [np.array([0] * len(visualizerData[0])) for _ in range(len(visualizerData))]
        freq_decay_counter = [1] * len(visualizerData[0])
        framesBeforeDecrease = round(self.settings.framerate / 6)  # 1/x of a second seems fine right?
        for i in range(len(visualizerData)):
            if i == 0:
                decay_data[i] = visualizerData[i]
                continue
            for freqIdx in range(len(visualizerData[i])):
                if (visualizerData[i][freqIdx] >= decay_data[i - 1][freqIdx]):
                    decay_data[i][freqIdx] = visualizerData[i][freqIdx]
                    freq_decay_counter[freqIdx] = 1
                else:
                    if (freq_decay_counter[freqIdx] > framesBeforeDecrease and decay_data[i - 1][
                        freqIdx] >= rate_of_decay):
                        decay_data[i][freqIdx] = decay_data[i - 1][freqIdx] + rate_of_decay
                    else:
                        decay_data[i][freqIdx] = decay_data[i - 1][freqIdx]
                    freq_decay_counter[freqIdx] += 1

        return np.array(decay_data)

    def createDrawables(self, fig, spectrumToShow, freqaxis, maxbarvalue):
        visAxisType = AxisType.CARTESIAN
        visShapeType = ShapeType.LINE
        if (self.settings.polar):
            visAxisType = AxisType.POLAR
        if (self.settings.bars):
            visShapeType = ShapeType.BAR

        # create the ghost drawable
        if (self.getAxis(fig, -10) != None):
            ghost_color = '#2F2F2F'
            ghostDecayRate = (-maxbarvalue) / self.settings.framerate
            ghost_decay_data = self.getGhostDecay(spectrumToShow, ghostDecayRate / 2)
            ghostArtistData = self.getArtistData(freqaxis, ghost_decay_data[0], self.getAxis(fig, -10), ghost_color, ghost_color)
            ghost_drawable = Drawable(ghostArtistData, ghost_decay_data, visAxisType, visShapeType, maxbarvalue)
            self.drawables.append(ghost_drawable)

        # create the main plot drawable
        artistData = self.getArtistData(freqaxis, spectrumToShow[0], self.getAxis(fig, 0), self.settings.bar_color,
                                   self.settings.bar_edge_color)
        visualizer_drawable = Drawable(artistData, spectrumToShow, visAxisType, visShapeType, maxbarvalue)
        self.drawables.append(visualizer_drawable)

        # create the background drawables, if necessary
        if (self.getAxis(fig, -99) != None):
            bkAxis = self.getAxis(fig, -99)
            bkArtists = []
            if (len(bkAxis.texts) > 0):
                bkArtists.append(bkAxis.texts)
            if (len(bkAxis.images) > 0):
                bkArtists.append(bkAxis.images)
            background_drawable = Drawable(bkArtists, None, AxisType.IMAGE, ShapeType.IMAGE)
            self.drawables.append(background_drawable)

    def getArtistData(self, freqaxis, firstFrameData, axis, fillColor, edgeColor):
        artistData = []
        if (self.settings.bars):
            artistData = self.getBarData(freqaxis, firstFrameData, axis, fillColor, edgeColor)
        else:
            artistData = self.getLineData(freqaxis, axis, edgeColor, fillColor)
            fillArtist = axis.fill_between(freqaxis, 0, firstFrameData, facecolor=fillColor)
            artistData.append(fillArtist)
        return artistData

    def getPercentageTicksArray(self, totalNumFrames):
        tenPercentTicks = []
        tenPercent = round(totalNumFrames / 10)
        for num in range(totalNumFrames):
            if (num != 0 and num % tenPercent == 0):
                tenPercentTicks.append(num)
        return tenPercentTicks

    def getLineData(self, freqaxis, ax, linecolor, fillcolor):
        line = ax.plot(freqaxis, color=linecolor)
        return line

    def getBarData(self, freqaxis, firstFrameData, ax, barcolor, baredgecolor):

        if (self.settings.polar):
            return ax.bar(freqaxis, firstFrameData, align='center', width=self.getBarWidth(len(freqaxis)), bottom=20,
                          color=barcolor,
                          linewidth=1, edgecolor=baredgecolor)
        else:
            return ax.bar(freqaxis, firstFrameData, align='center', width=self.getBarWidth(), color=barcolor, linewidth=1,
                          edgecolor=baredgecolor, zorder=ax.zorder)

    def getBarWidth(self, axisLength=0):
        # 75 width at 80 bars is perfect. These aren't scientific numbers, but they're a nice baseline.
        # TODO: make this look not-bad at other resolutions
        defaultWidth = 75
        defaultBars = 80
        barwidth = (defaultWidth) * (defaultBars / self.settings.number_of_points)
        if (self.settings.polar):
            barwidth = 2 * np.pi / axisLength
        return barwidth

    def initAnimation(self):
        artists = []
        for item in self.drawables:
            if (item.shape_type == ShapeType.BAR):
                artists.extend(list(item.artist_info))
            if (item.shape_type == ShapeType.LINE):
                artists.extend(list(item.artist_info))
            if (item.axis_type == AxisType.IMAGE):
                for artist in item.artist_info:
                    artists.extend(list(artist))

        return artists

    def animate(self, frame, freqaxis, progress, drawables):

        artists = []
        for item in drawables:
            if (item.shape_type == ShapeType.BAR):
                self.draw_bars(item, frame)
                artists.extend(list(item.artist_info))
            if (item.shape_type == ShapeType.LINE):
                self.draw_line(item, freqaxis, frame)
                artists.extend(list(item.artist_info))

        ########################################
        if frame in progress:
            progress.pop(0)
            logging.info('Animation ' + str(100 - (len(progress) * 10)) + '% complete')
        return artists

    def draw_bars(self, drawable, frame_num):
        for i in range(len(drawable.data[frame_num])):
            drawable.artist_info[i].set_height(drawable.data[frame_num][i])

    def draw_line(self, drawable, freqaxis, frame_num):

        drawable.artist_info[0].set_data(freqaxis, drawable.data[frame_num])

        drawable.artist_info[1].axes.collections.clear()
        fillcolor = self.settings.bar_color
        if (drawable.artist_info[1].axes.zorder == -10):
            fillcolor = '#2F2F2F'
        drawable.artist_info[1].axes.fill_between(freqaxis, 0, drawable.data[frame_num], facecolor=fillcolor)

    def getPolarValue(self, value, maxValue, minValue=10):
        slope = (maxValue - minValue) / (maxValue)
        return slope * (value) + minValue

    def configurePlot(self):
        if (self.settings.polar):
            fig, visualizer_axis = pyplot.subplots(subplot_kw=dict(projection="polar"))
            self.cleanPolarPlot(visualizer_axis)
        else:
            fig, visualizer_axis = pyplot.subplots()
            self.cleanCartesianPlot(visualizer_axis)

        if (self.settings.ghost):
            self.addGhostVisualizer(fig)
        fig.set_size_inches(self.settings.resolution_x / self.settings.dpiMultiplier,
                            self.settings.resolution_y / self.settings.dpiMultiplier)

        self.setAxis(visualizer_axis)

        self.createBackground(fig, visualizer_axis)
        if (self.settings.text != ''):
            self.addText(fig)

        return fig

    def setAxis(self, axis):

        axis.set_ylim([0, 80])
        if (not self.settings.bars) and (not self.settings.polar):
            axis.set_xlim([0, self.settings.max_frequency])

    def cleanPolarPlot(self, axis):
        pyplot.axis('off')
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)

    def cleanCartesianPlot(self, axis):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

    def createBackground(self, fig, ax):
        if (self.settings.bk_img_path != ''):
            logging.info('!!! IMAGE SPECIFIED, THIS WILL TAKE A LOT LONGER TO RENDER, SORRY!!!')
            img = pyplot.imread(self.settings.bk_img_path)
            ax_image = self.createBackgroundAxisIfNotExists(fig)
            self.cleanCartesianPlot(ax_image)
            ax_image.imshow(img, extent=(0, self.settings.resolution_x, 0, self.settings.resolution_y), resample=False)
            fig.patch.set_visible(False)
        elif (self.settings.bk_img_color != ''):
            ax_image = self.createBackgroundAxisIfNotExists(fig)
            ax_image.set_facecolor(self.settings.bk_img_color)
            fig.set_facecolor(self.settings.bk_img_color)
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

    def addText(self, fig):
        ax_image = self.createBackgroundAxisIfNotExists(fig)
        message = ax_image.text(0.5, 0.05, self.settings.text, fontsize='50', horizontalalignment='center',
                                verticalalignment='center',
                                color=self.settings.text_color,
                                transform=ax_image.transAxes)
        if (self.settings.use_text_outline):
            message.set_path_effects(
                [path_effects.Stroke(linewidth=self.settings.text_outline_width, foreground=self.settings.text_outline_color),
                 path_effects.Normal()])

    def addGhostVisualizer(self, fig):
        if (self.settings.polar):
            ghost_axis = fig.add_subplot(polar=True, zorder=-10)
            self.cleanPolarPlot(ghost_axis)
        else:
            ghost_axis = fig.add_subplot(zorder=-10)
            self.cleanCartesianPlot(ghost_axis)
        self.setAxis(ghost_axis)
        ghost_axis.patch.set_visible(False)

    def createBackgroundAxisIfNotExists(self, fig):
        for axis in fig.axes:
            if axis.zorder == -99:
                return axis
        return fig.add_axes([0, 0, 1, 1], label="background", zorder=-99)

    def getAxis(self, fig, zorder):
        return next((axis for axis in fig.axes if axis.zorder == zorder), None)

