import numpy as np
from base.base_filter import Filter

class InvertFilter(Filter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        return 255 - image
