import json
from settings import Settings
from visualizer import Visualizer

if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data

    #the gui needs to create a new config file. the code that runs this will eventually be it's own file.
    with open('./sample.json') as s:
        samplesettings = json.load(s)
    demoVis = Visualizer(Settings(samplesettings))
    #demoVis.generate_visualizer()
    demoVis.show_gui()

    #with open('./config.json') as f:
    #    json_settings = json.load(f)
    #settings = Settings(json_settings)
    #visualizer = Visualizer(settings)
    #visualizer.generate_visualizer()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
