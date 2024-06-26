import numpy as np
from base.base_filter import Filter
from image_utils import denormalize, normalize

class SaturationFilter(Filter):
    def __init__(self, saturation):
        self.saturation = saturation

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image  # Cant be use on grayscale images
        image = normalize(image)
        gray = np.dot(image, [0.2989, 0.5870, 0.1140])
        gray = np.expand_dims(gray, axis=2)
        image = gray + (image - gray) * (self.saturation + 1)
        return denormalize(image)
