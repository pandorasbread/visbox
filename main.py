import json
from settings import Settings
from visualizer import Visualizer

if __name__ == '__main__':
    # eventually pull in JSON object with wav, layout, and text data
    with open('./config.json') as f:
        json_settings = json.load(f)

    #demoVis = Visualizer()

    settings = Settings(json_settings)
    visualizer = Visualizer(settings)
    visualizer.generate_visualizer()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
