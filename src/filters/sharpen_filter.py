import numpy as np
from base.base_filter import Filter
from image_utils import denormalize, normalize
from .sobel_filter import SobelFilter

class SharpenFilter(Filter):
    def __init__(self, magnitude=1):
        self.magnitude = magnitude
        self.sobel_filter = SobelFilter()

    def apply(self, image: np.ndarray, edge_image: np.ndarray = None) -> np.ndarray:
        if edge_image is None:
            edge_image = self.sobel_filter.apply(image)

        edge_image = normalize(edge_image)
        sharpened_edges = edge_image * self.magnitude
        input_image_float32 = normalize(image)
        sharpened_image = input_image_float32 + sharpened_edges

        return denormalize(sharpened_image)
