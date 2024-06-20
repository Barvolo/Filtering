from abc import ABC, abstractmethod
import numpy as np

class Filter(ABC):
    """
    Abstract base class for image filters.
    """
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _denormalize(image: np.ndarray) -> np.ndarray:
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
