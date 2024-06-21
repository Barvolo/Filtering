import numpy as np
from base.base_filter import Filter

class BoxBlur(Filter):
    """
    GEN: Asked ChatGPT for help with an overflow problem in the box blur implementation.
    Prompt: "I'm implementing a box blur filter in Python using NumPy, but I'm
    encountering overflow issues with large images or large blur kernels. 
    What approaches can I use to solve this problem? Here's my current implementation:
    [inserted code snippet here]"

    The solution involves using float64 data type for intermediate calculations
    to prevent overflow, and an integral image approach for efficiency.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def compute_integral_image(self, image):
        # Using float64 to prevent overflow on large images
        integral_image = np.cumsum(np.cumsum(image.astype(np.float64), axis=0), axis=1)
        return integral_image
    
    def apply(self, image: np.ndarray) -> np.ndarray:
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

        # Ensure output is clipped to valid range and converted back to uint8
        blurred_image = np.clip(blurred_image, 0, 255)
        blurred_image_uint8 = blurred_image.astype(np.uint8)

        return blurred_image_uint8 if blurred_image.shape[2] > 1 else blurred_image_uint8[:, :, 0]
