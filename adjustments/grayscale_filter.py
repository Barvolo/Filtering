import numpy as np
from filters.base_filter import Filter

class GrayscaleFilter(Filter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:  # Already grayscale
            return image
        # Convert RGB to Grayscale using luminosity method
        grayscale_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return grayscale_image.astype(np.uint8)
