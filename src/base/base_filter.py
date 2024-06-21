from abc import ABC, abstractmethod
import numpy as np

class Filter(ABC):
    """
    Abstract base class for image filters.
    """
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    