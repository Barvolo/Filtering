import numpy as np
from base.base_filter import Filter

class SobelFilter(Filter):
    def __init__(self):
        self.Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        self.Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


    def apply(self, image: np.ndarray) -> np.ndarray:
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
        """
        GEN: ChatGPT assisted in selecting the padding mode for image edges.
        Prompt: "What are the options for handling the edges of an image when applying padding, and which is best for maintaining edge continuity?"
        """
        padded_image = np.pad(image, 1, mode='edge')
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
