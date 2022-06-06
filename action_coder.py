import numpy as np

class ActionCoder:
    def __init__(self, resolution):
        self.idx = [[x, y, 0] for x in np.arange(-1, 1, resolution) for y in np.arange(-1, 1, resolution)]

    def convert(self, id):
        return self.idx[id]