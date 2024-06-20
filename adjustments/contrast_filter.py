import numpy as np
from filters.base_filter import Filter

class ContrastFilter(Filter):
    def __init__(self, contrast):
        self.contrast = contrast

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self._normalize(image)
        mean = np.mean(image)
        image = (image - mean) * (self.contrast + 1) + mean
        return self._denormalize(image)
