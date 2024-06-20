import numpy as np
from filters.base_filter import Filter

class BrightnessFilter(Filter):
    def __init__(self, brightness):
        self.brightness = brightness

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self._normalize(image)
        image = image + self.brightness / 255.0
        return self._denormalize(image)
