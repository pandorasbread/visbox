import tkinter as tk
from enum import Enum
import re

import PySimpleGUI as sg
from settings import Settings

class Gui:
    def __init__(self, settings: Settings):
        sg.theme('Material 1')
        self.settings = settings
        self.window = sg.Window('Visbox', self.get_layout(), finalize=True)

        self.audiofilepath = self.window[EventType.AUDIO_FILE_PATH]
        self.numbars = self.window[EventType.NUMBER_OF_BARS]
        self.polar = self.window[EventType.POLAR]
        self.ghost = self.window[EventType.GHOST]
        self.barscale = self.window[EventType.BAR_SCALE]
        self.maxfreq = self.window[EventType.MAX_FREQ]
        self.bkimg = self.window[EventType.BK_IMG_PATH]
        self.text = self.window[EventType.TEXT]
        self.usetextoutline = self.window[EventType.USE_TEXT_OUTLINE]
        self.textoutlinewidth = self.window[EventType.TEXT_OUTLINE_WIDTH]
        self.framerate = self.window[EventType.FRAMERATE]
        self.resolution = self.window[EventType.RESOLUTION_TYPE]
        self.fillcolor = self.window[EventType.FILL_COLOR]
        self.edgecolor = self.window[EventType.EDGE_COLOR]
        self.textcolor = self.window[EventType.TEXT_COLOR]
        self.textoutlinecolor = self.window[EventType.TEXT_OUTLINE_COLOR]
        self.bkcolor = self.window[EventType.BK_COLOR]
        self.useedgecolor = self.window[EventType.USE_EDGE_COLOR]
        self.color = None


    def get_layout(self):
        LABEL_PADDING = ((0,0),(15,0))

        return [
            [sg.Text('Visbox V.Dev.Alpha.4 (now with GUI!)', size=(42,1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
            [
                sg.Text('Audio File', size=(15, 1), auto_size_text=False, justification='right'),
                sg.InputText(self.settings.audio_file_path, disabled=True, enable_events=True, key=EventType.AUDIO_FILE_PATH), sg.FileBrowse(size=(15,1), file_types=(("wav files", "*.wav"), ("MP3 files", "*.mp3")))
            ],
            [sg.Frame(layout=[
                [sg.Checkbox('Ghost', size=(17, 1), key=EventType.GHOST, default=self.settings.ghost, enable_events=True), sg.Text('Max Frequency', size=(12,1), p=LABEL_PADDING), sg.Text('Frame Rate (in fps)', size=(15, 1), p=LABEL_PADDING)],
                [sg.Checkbox('Circular Visualizer', default=self.settings.polar, size=(17, 1), key=EventType.POLAR, enable_events=True), sg.InputText(self.settings.max_frequency, key=EventType.MAX_FREQ, size=(5, 1), enable_events=True), sg.Text('', size=(5,1)),sg.InputText(self.settings.framerate, key=EventType.FRAMERATE, size=(5, 1), enable_events=True)],

                [sg.Text('', size=(11,1)), sg.Text('# of Bars', size=(8,1), p=LABEL_PADDING), sg.Text('Bar Scale', p=LABEL_PADDING)],
                [sg.Radio('Bars', "barsorlines", default=True, enable_events=True, size=(8, 1), key=EventType.BARS), sg.InputText(self.settings.number_of_points, size=(5,1), key=EventType.NUMBER_OF_BARS, enable_events=True), sg.Text('', size=(1,1)), sg.InputText(self.settings.bar_scale, key=EventType.BAR_SCALE, size=(5,1), enable_events=True)],
                [sg.Radio('Line', "barsorlines", key=EventType.LINE, enable_events=True)],

                [sg.Text('Fill Color', size=(12, 1), p=LABEL_PADDING), sg.Text('Edge Color', size=(10, 1), p=LABEL_PADDING)],
                [sg.Button('', size=(10, 1), button_color=('#FFFFFF', self.settings.bar_color), key=EventType.FILL_COLOR), sg.Button('', size=(10, 1), button_color=('#FFFFFF', self.settings.bar_edge_color), key=EventType.EDGE_COLOR, disabled=self.settings.bar_edge_color == self.settings.bar_color), sg.Checkbox('Use Edge Color', default=self.settings.bar_edge_color != self.settings.bar_color, key=EventType.USE_EDGE_COLOR, size=(15,1), enable_events=True)]],
                title='Visualizer Settings', relief=sg.RELIEF_SUNKEN,
                tooltip='Use these to set flags', size=(400,250)),
                sg.Frame(layout=[
                    [sg.Text('Text to Show', size=(19,1), p=LABEL_PADDING, enable_events=True),sg.Text('Text Color', size=(10, 1), p=LABEL_PADDING)],
                    [sg.InputText(self.settings.text, key=EventType.TEXT, size=(20, 1)), sg.Button('', size=(10, 1), button_color=('#FFFFFF', self.settings.text_color), key=EventType.TEXT_COLOR)],
                    [sg.Text('', size=(18,1)), sg.Text('Text Outline Color', size=(14,1), p=LABEL_PADDING), sg.Text('Text Outline Width', size=(17,1), p=LABEL_PADDING)],
                    [sg.Checkbox('Use Text Outline', default=self.settings.use_text_outline, key=EventType.USE_TEXT_OUTLINE, size=(15,1), enable_events=True),
                     sg.Button('', size=(10, 1), button_color=('#000000', self.settings.text_outline_color), key=EventType.TEXT_OUTLINE_COLOR),
                     sg.Text('', size=(1, 1)),
                     sg.InputText(self.settings.text_outline_width, key=EventType.TEXT_OUTLINE_WIDTH, size=(5, 1), enable_events=True), sg.Text('', size=(10,1))],
                    ],
                    title='Text Settings', relief=sg.RELIEF_SUNKEN,
                    tooltip='Use these to set flags', size=(400,250)),
            ],
            [sg.Frame(layout=[
                    [sg.Text('If you specify both an image and a color,')],
                    [sg.Text('only an image will be included.')],
                    [sg.Text('Background Color', size=(15, 1), p=LABEL_PADDING), sg.Text('Background Image', size=(15, 1), p=LABEL_PADDING)],
                    [sg.Button('', size=(10, 1), button_color=('#FFFFFF', self.settings.bk_img_color), key=EventType.BK_COLOR), sg.Text('', size=(1,1)),
                    sg.InputText(self.settings.bk_img_path, size=(15,1), disabled=True, key=EventType.BK_IMG_PATH, enable_events=True),
                    sg.FileBrowse(size=(10, 1), file_types=(("jpg files", "*.jpg"), ("jpeg files", "*.jpeg"), ("png files", "*.png"))),
                    sg.Button('Clear', size=(10, 1), key=EventType.CLEAR_IMAGE)]
                ],
                    title='Background Settings', relief=sg.RELIEF_SUNKEN, size=(400, 150))],
            [sg.Frame(layout=[
               [sg.Text('Resolution', size=(12, 1), p=LABEL_PADDING), ],
               [sg.InputCombo(key=EventType.RESOLUTION_TYPE, values=('720p', 'twitter', '1080p', '4k'), size=(10, 1), default_value=self.settings.resolution_type, enable_events=True)]],
               title='Video Settings', relief=sg.RELIEF_SUNKEN)],
            [sg.Submit(key='Exit')],
        ]

    def handle_events(self):
        while True:
            event, values = self.window.read()

            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break
            elif event == EventType.AUDIO_FILE_PATH:
                continue
            elif event == EventType.NUMBER_OF_BARS:
                self.updateIntField(self.numbars, values[EventType.NUMBER_OF_BARS])
            elif event == EventType.POLAR:
                continue
            elif event == EventType.GHOST:
                continue
            elif event == EventType.BAR_SCALE:
                self.updateDecimalField(self.barscale, values[EventType.BAR_SCALE])
            elif event == EventType.MAX_FREQ:
                self.updateIntField(self.maxfreq, values[EventType.MAX_FREQ])
            elif event == EventType.BK_IMG_PATH:
                continue
            elif event == EventType.CLEAR_IMAGE:
                self.bkimg.Update('')
            elif event == EventType.TEXT:
                continue
            elif event == EventType.USE_TEXT_OUTLINE:
                self.textoutlinecolor.Update(disabled=not values[EventType.USE_TEXT_OUTLINE])
                self.textoutlinewidth.Update(disabled=not values[EventType.USE_TEXT_OUTLINE])
            elif event == EventType.TEXT_OUTLINE_WIDTH:
                self.updateIntField(self.textoutlinewidth, values[EventType.TEXT_OUTLINE_WIDTH])
            elif event == EventType.FRAMERATE:
                self.updateIntField(self.framerate, values[EventType.FRAMERATE])
            elif event == EventType.RESOLUTION_TYPE:
                continue
            elif event == EventType.FILL_COLOR:
                self.updateColorPicker(self.fillcolor)
                if (not values[EventType.USE_EDGE_COLOR]):
                    self.edgecolor.Update(button_color=(self.fillcolor.ButtonColor[1],self.fillcolor.ButtonColor[1]))
            elif event == EventType.USE_EDGE_COLOR:
                if (not values[EventType.USE_EDGE_COLOR]):
                    self.edgecolor.Update(button_color=(self.fillcolor.ButtonColor[1],self.fillcolor.ButtonColor[1]))
                    self.edgecolor.Update(disabled=True)
                else:
                    self.edgecolor.Update(disabled=False)

            elif event == EventType.EDGE_COLOR:
                self.updateColorPicker(self.edgecolor)
            elif event == EventType.TEXT_COLOR:
                self.updateColorPicker(self.textcolor)
            elif event == EventType.TEXT_OUTLINE_COLOR:
                self.updateColorPicker(self.textoutlinecolor)
            elif event == EventType.BK_COLOR:
                self.updateColorPicker(self.bkcolor)
            elif event == EventType.BARS:
                self.barscale.Update(disabled=False)
                self.numbars.Update(disabled=False)
                self.numbars.Update(value=80)
            elif event == EventType.LINE:
                self.barscale.Update(disabled=True)
                self.numbars.Update(disabled=True)
                self.numbars.Update(value=0)

        self.updateSettings(values)
        #write a new/update a new/send a json file here or something
        self.window.close()
        return self.settings


    def updateColorPicker(self, picker):
        colors = tk.colorchooser.askcolor(
            parent=picker.ParentForm.TKroot, color=picker.ButtonColor[1])
        color = colors[1]
        picker.Update(button_color=(color, color))

    def updateIntField(self, field, value):
            field.Update(re.sub("[^0-9]", "", value))

    def updateDecimalField(self, field, value):
            field.Update(re.sub("[^\d*\.?\d+$]", "", value))

    def updateSettings(self, values):

        self.settings.audio_file_path = values[EventType.AUDIO_FILE_PATH]
        self.settings.number_of_points = int(values[EventType.NUMBER_OF_BARS])
        self.settings.polar = values[EventType.POLAR]
        self.settings.ghost = values[EventType.GHOST]
        self.settings.bar_scale = float(values[EventType.BAR_SCALE].strip('"'))
        self.settings.max_frequency = int(values[EventType.MAX_FREQ])
        self.settings.bk_img_path = values[EventType.BK_IMG_PATH]
        self.settings.text = values[EventType.TEXT]
        self.settings.use_text_outline = values[EventType.USE_TEXT_OUTLINE]
        self.settings.text_outline_width = values[EventType.TEXT_OUTLINE_WIDTH]
        self.settings.framerate = int(values[EventType.FRAMERATE])
        self.settings.resolution_type = values[EventType.RESOLUTION_TYPE]
        self.settings.resolution_x, self.settings.resolution_y = self.settings.getxyResolutions(values[EventType.RESOLUTION_TYPE])
        self.settings.bar_color = self.fillcolor.ButtonColor[1]
        self.settings.bar_edge_color = self.edgecolor.ButtonColor[1]
        self.settings.text_color = self.textcolor.ButtonColor[1]
        self.settings.text_outline_color = self.textoutlinecolor.ButtonColor[1]
        self.settings.bk_img_color = self.bkcolor.ButtonColor[1]
        self.settings.saveToJSON()

class EventType(Enum):
    AUDIO_FILE_PATH = "audio_file_path"
    NUMBER_OF_BARS = "number_of_bars"
    POLAR = "polar"
    GHOST = "ghost"
    BARS = "bars"
    LINE = "line"
    BAR_SCALE = "bar_scale"
    FILL_COLOR = "bar_color"
    EDGE_COLOR = "bar_edge_color"
    MAX_FREQ = "max_frequency"
    BK_IMG_PATH = "background_img_path"
    BK_COLOR = "background_color"
    TEXT = "text"
    TEXT_COLOR = "text_color"
    USE_TEXT_OUTLINE = "use_text_outline"
    TEXT_OUTLINE_COLOR = "text_outline_color"
    TEXT_OUTLINE_WIDTH = "text_outline_width"
    FRAMERATE = "framerate"
    RESOLUTION_TYPE = "resolution"
    USE_EDGE_COLOR = "use_edge_color"
    CLEAR_IMAGE = "clear_image"

