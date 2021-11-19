import json

class Settings:
    def __init__(self, data):
        self.number_of_points = data["number_of_bars"]
        self.polar = data["polar"]
        self.bars = True
        if (self.number_of_points == 0):
            self.bars = False
            self.number_of_points = 80
        # We need to do hacky shit to have a default amount of 80 line points.. ehhghhhhh

        # 10k is pretty good for straight bars, 8k looks good in a circle
        self.max_frequency = data["max_frequency"]
        self.bar_color = data["bar_color"]
        if self.bar_color == '':
            self.bar_color = '#000000'
        self.bar_edge_color = data["bar_edge_color"]
        if (self.bar_edge_color == ""):
            self.bar_edge_color = self.bar_color
        self.bar_scale = data["bar_scale"]
        self.framerate = data["framerate"]
        self.bk_img_path = data["background_img_path"]
        self.bk_img_color = data["background_color"]
        self.audio_file_path = data["audio_file_path"]
        self.resolution_type = data["resolution"]
        self.resolution_x, self.resolution_y = self.getxyResolutions(self.resolution_type)

        self.text = data["text"]
        self.text_color = data["text_color"]
        if (self.text_color == ''):
            self.text_color = '#FFFFFF'
        self.use_text_outline = data["use_text_outline"]
        self.text_outline_color = data["text_outline_color"]
        if (self.text_outline_color == ''):
            self.text_outline_color = self.text_color
        self.text_outline_width = data["text_outline_width"]
        self.ghost = data["ghost"]
        self.dpiMultiplier = 100
        self.demo = False

    def getxyResolutions(self, resolutionOption):
        if (resolutionOption == "4k"):
            return 3840, 2160
        elif (resolutionOption == "1080p"):
            return 1920, 1080
        elif (resolutionOption == "twitter"):
            return 1000, 1000
        elif (resolutionOption == "720p"):
            return 1280, 720
        else:
            return 1000, 1000

    def saveToJSON(self):
        toSave = {
            "audio_file_path": self.audio_file_path,
            "number_of_bars": self.number_of_points,
            "polar": self.polar,
            "ghost": self.ghost,
            "bar_scale": self.bar_scale,
            "bar_color": self.bar_color,
            "bar_edge_color": self.bar_edge_color,
            "max_frequency": self.max_frequency,
            "background_img_path": self.bk_img_path,
            "background_color": self.bk_img_color,
            "text": self.text,
            "text_color": self.text_color,
            "use_text_outline": self.use_text_outline,
            "text_outline_color": self.text_outline_color,
            "text_outline_width": self.text_outline_width,
            "framerate": self.framerate,
            "resolution": self.resolution_type
        }

        toSaveJson = json.dumps(toSave)
        with open('config.json', 'w') as outfile:
            json.dump(toSave, outfile)
