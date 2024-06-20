import numpy as np
from .base_filter import Filter

class BoxBlur(Filter):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def compute_integral_image(self, image):
        integral_image = np.cumsum(np.cumsum(image.astype(np.float64), axis=0), axis=1)
        return integral_image
    
    def apply(self, image):
        if image.ndim == 2:
            image = image[:, :, np.newaxis] 

        blurred_image = np.zeros_like(image, dtype=np.float64)
        integral_image = self.compute_integral_image(image)
        image_height, image_width, num_channels = image.shape

        for y in range(image_height):
            for x in range(image_width):
                y1 = max(0, y - self.height // 2)
                y2 = min(image_height, y + self.height // 2 + 1)
                x1 = max(0, x - self.width // 2)
                x2 = min(image_width, x + self.width // 2 + 1)

                for c in range(num_channels):
                    total = integral_image[y2-1, x2-1, c]
                    if y1 > 0:
                        total -= integral_image[y1-1, x2-1, c]
                    if x1 > 0:
                        total -= integral_image[y2-1, x1-1, c]
                    if y1 > 0 and x1 > 0:
                        total += integral_image[y1-1, x1-1, c]

                    num_pixels = (y2 - y1) * (x2 - x1)
                    blurred_image[y, x, c] = total / num_pixels

        blurred_image = np.clip(blurred_image, 0, 255)
        blurred_image_uint8 = blurred_image.astype(np.uint8)

        return blurred_image_uint8 if blurred_image.shape[2] > 1 else blurred_image_uint8[:, :, 0]
