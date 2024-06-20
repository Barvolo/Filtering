from filters.base_filter import Filter
from filters.sharpen_filter import SharpenFilter
from filters.sobel_filter import SobelFilter

class ImageProcessor:
    def __init__(self):
        self.filters = []

    def add_filter(self, filter):
        if not issubclass(type(filter), Filter):
            raise TypeError("Expected object of type Filter")
        self.filters.append(filter)

    def process_image(self, image):
        processed_image = image
        sobel_edge_image = None
        for filter in self.filters:
            if isinstance(filter, SharpenFilter):
                processed_image = filter.apply(processed_image, sobel_edge_image)
            else:
                processed_image = filter.apply(processed_image)
                if isinstance(filter, SobelFilter):
                    sobel_edge_image = processed_image
        return processed_image
