import json

from gui import Gui
from settings import Settings
from visualizer import Visualizer
from os.path import exists

if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data

    #the gui needs to create a new config file. the code that runs this will eventually be it's own file.
    jsonfilename = './sample.json'
    if (exists('./config.json')):
        jsonfilename = './config.json'

    with open(jsonfilename) as s:
        samplesettings = json.load(s)

    gui = Gui(Settings(samplesettings))
    gui.handle_events()
    #demoVis = Visualizer(Settings(samplesettings))
    #newsettings = demoVis.show_gui()

    #visualizer = Visualizer(newsettings)
    #visualizer.generate_visualizer()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
