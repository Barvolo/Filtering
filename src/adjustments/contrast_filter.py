import numpy as np
from base.base_filter import Filter
from image_utils import denormalize, normalize

class ContrastFilter(Filter):
    def __init__(self, contrast):
        self.contrast = contrast

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = normalize(image)
        mean = np.mean(image)
        image = (image - mean) * (self.contrast + 1) + mean
        return denormalize(image)
