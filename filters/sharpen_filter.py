import numpy as np
from .base_filter import Filter
from .sobel_filter import SobelFilter

class SharpenFilter(Filter):
    def __init__(self, magnitude=1):
        self.magnitude = magnitude
        self.sobel_filter = SobelFilter()

    def apply(self, image, edge_image=None):
        if edge_image is None:
            edge_image = self.sobel_filter.apply(image)

        edge_image = self._normalize(edge_image)
        sharpened_edges = edge_image * self.magnitude
        input_image_float32 = self._normalize(image)
        sharpened_image = input_image_float32 + sharpened_edges

        return self._denormalize(sharpened_image)
