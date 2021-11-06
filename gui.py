import tkinter as tk
from enum import Enum

import PySimpleGUI as sg
from settings import Settings

class Gui:
    def __init__(self, settings: Settings):
        sg.theme('Material 2')
        self.settings = settings
        self.window = sg.Window('Visbox', self.get_layout(), finalize=True)

        self.chooser = self.window[EventType.FILL_COLOR]
        self.color = None


    def get_layout(self):
       return [
            [sg.Text('Visbox V.Dev.Alpha.4 (now with GUI!)', size=(30,1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
            [
                sg.Text('Audio File', size=(15, 1), auto_size_text=False, justification='right'),
                sg.InputText(self.settings.audio_file_path), sg.FileBrowse(size=(15,1), file_types=(("wav files", "*.wav"), ("MP3 files", "*.mp3")))
            ],
            [sg.Frame(layout=[
               [sg.Checkbox('Ghost', size=(10, 1), key=EventType.GHOST, default=True), sg.Checkbox('Circular Visualizer', default=True, key=EventType.POLAR)],
               [sg.Radio('Bars', "barsorlines", default=True, size=(10, 1), key=EventType.BARS), sg.Radio('Line', "barsorlines", key=EventType.BARS)]],
                title='Graph Settings', relief=sg.RELIEF_SUNKEN,
               tooltip='Use these to set flags')],
            [sg.Button("Fill Color", size=(1, 1), button_color=('#1f77b4', '#1f77b4'), key=EventType.FILL_COLOR)],
            [sg.Submit(key='Exit')],
        ]

    def handle_events(self):
        while True:
            event, values = self.window.read()

            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break
            elif event == EventType.FILL_COLOR:
                colors = tk.colorchooser.askcolor(
                        parent=self.chooser.ParentForm.TKroot, color=self.color)
                color = colors[1]
                self.chooser.Update(button_color=(color, color))

        #write a new/update a new/send a json file here or something
        self.window.close()

class EventType(Enum):
    GHOST = "GHOST"
    POLAR = "POLAR"
    BARS = "BARS"
    MAX_FREQ = "MAX_FREQ"
    FILL_COLOR = "FILL_COLOR"
    EDGE_COLOR = "EDGE_COLOR"
    BAR_SCALE = "BAR_SCALE"
    FRAMERATE = "FRAMERATE"
    BK_IMG_PATH = "BK_IMG_PATH"
    BK_COLOR = "BK_COLOR"
    AUDIO_FILE_PATH = "AUDIO_FILE_PATH"
    RESOLUTION_TYPE = "RESOLUTION_TYPE"
    TEXT = "TEXT"
    TEXT_COLOR = "TEXT_COLOR"
    USE_TEXT_OUTLINE = "USE_TEXT_OUTLINE"
    TEXT_OUTLINE_COLOR = "TEXT_OUTLINE_COLOR"
    TEXT_OUTLINE_WIDTH = "TEXT_OUTLINE_WIDTH"
