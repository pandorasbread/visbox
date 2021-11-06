class Settings:

    #def __init__(self):
        #load the most basic data
        #self.number_of_points = 80
        #self.polar = True
        #self.bars = True
        # 10k is pretty good for straight bars, 8k looks good in a circle
        #self.max_frequency = 10000
        #self.bar_color = '#000000'
        #self.bar_edge_color = self.bar_color
        #self.bar_scale = 1.0
        #self.framerate = 30
        #self.bk_img_path = ''
        #self.bk_img_color = '#FFFFFF'
        #self.audio_file_path = 'visboxsample.wav'
        #self.resolution_type = 'twitter'
        #self.resolution_x, self.resolution_y = self.getxyResolutions(self.resolution_type)
        #self.text = ''
        #self.text_color = '#FFFFFF'
        #self.use_text_outline = True
        #self.text_outline_color = '#000000'
        #self.text_outline_width = 2
        #self.ghost = True
        #self.dpiMultiplier = 100
        #self.demo = True

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