import numpy as np

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the input image to the range [0, 1].
    """
    return image.astype(np.float32) / 255.0

def denormalize(image: np.ndarray) -> np.ndarray:
    """
    Denormalize the input image to the range [0, 255].
    """
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)