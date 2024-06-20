import numpy as np
from .base_filter import Filter

class SobelFilter(Filter):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'Gx'):
            self.Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
            self.Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    def apply(self, image):
        if image.ndim == 2:  # Grayscale image
            return self._apply_sobel(image)
        elif image.ndim == 3:  # RGB image
            return self._apply_sobel_rgb(image)
        else:
            raise ValueError("Unsupported image format")
        
    def _compute_gradient(self, region):
        gx = np.sum(region * self.Gx)
        gy = np.sum(region * self.Gy)
        gradient = np.sqrt(gx**2 + gy**2)
        return gradient

    def _apply_sobel(self, image):
        padded_image = np.pad(image, 1, mode='constant', constant_values=0)
        height, width = image.shape
        sobel_image = np.zeros_like(image, dtype=np.float32)

        for i in range(height):
            for j in range(width):
                region = padded_image[i:i+3, j:j+3]
                sobel_image[i, j] = self._compute_gradient(region)
                

        return np.clip(sobel_image, 0, 255).astype(np.uint8)

    def _apply_sobel_rgb(self, image):
        sobel_image = np.zeros_like(image, dtype=np.float32)
        for c in range(3):  
            sobel_image[:, :, c] = self._apply_sobel(image[:, :, c])
        return np.clip(sobel_image, 0, 255).astype(np.uint8)
