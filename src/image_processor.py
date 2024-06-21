from base.base_filter import Filter
from filters.sharpen_filter import SharpenFilter
from filters.sobel_filter import SobelFilter

class ImageProcessor:
    """
    GEN: ChatGPT provided guidance on designing a flexible image processing pipeline.
    Prompt: "How to design a Python class that can apply multiple image filters in sequence?
    I need to support filters like brightness, contrast, and box blur, applied in the order specified by the user."
    """
    
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
