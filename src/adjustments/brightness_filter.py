import numpy as np
from base.base_filter import Filter
from image_utils import denormalize, normalize


class BrightnessFilter(Filter):
    def __init__(self, brightness):
        self.brightness = brightness

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = normalize(image)
        image = image + self.brightness / 255.0
        return denormalize(image)
