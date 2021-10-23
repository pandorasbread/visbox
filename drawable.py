from enum import Enum
class Drawable:
    axis_type = ""
    shape_type = ""
    data = []
    max_value = 0

    def __init__(self, artist, dat, axisType, graphType, maxvalue=0):
        self.data = dat
        self.artist_info = artist
        self.axis_type = axisType
        self.shape_type = graphType
        self.max_value = maxvalue


class AxisType(Enum):
    POLAR = "POLAR"
    CARTESIAN = "CARTESIAN"
    IMAGE = "IMAGE"

class ShapeType(Enum):
    LINE = "LINE"
    BAR = "BAR"
    IMAGE = "BK"