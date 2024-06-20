import argparse
from filters.box_blur import BoxBlur
from filters.sobel_filter import SobelFilter
from filters.sharpen_filter import SharpenFilter
from adjustments.brightness_filter import BrightnessFilter
from adjustments.contrast_filter import ContrastFilter
from adjustments.saturation_filter import SaturationFilter
from adjustments.grayscale_filter import GrayscaleFilter
from adjustments.invert_filter import InvertFilter

def parse_args():
    parser = argparse.ArgumentParser(description='Apply filters to an image.')
    parser.add_argument('--input', required=True, help='Path to the input image')
    parser.add_argument('--output', help='Path to save the output image')
    parser.add_argument('--display', action='store_true', help='Display the output image')
    parser.add_argument('--filters', required=True, nargs='+', help='List of filters to apply with optional parameters, e.g., box:width=5,height=5 sobel sharpen:magnitude=1.5 brightness:brightness=20 contrast:contrast=1.5 saturation:saturation=1.2 grayscale invert sepia')

    return parser.parse_args()

def parse_filters(filter_args):
    filters = []
    for filter_arg in filter_args:
        try:
            filter_parts = filter_arg.split(':')
            filter_name = filter_parts[0]
            filter_params = {}

            if len(filter_parts) > 1:
                param_parts = filter_parts[1].split(',')
                for param in param_parts:
                    key, value = param.split('=')
                    filter_params[key] = value

            if filter_name == 'box':
                width = int(filter_params.get('width', 5))
                height = int(filter_params.get('height', 5))
                filters.append(BoxBlur(width, height))
            elif filter_name == 'sobel':
                filters.append(SobelFilter())
            elif filter_name == 'sharpen':
                magnitude = float(filter_params.get('magnitude', 1.0))
                filters.append(SharpenFilter(magnitude))
            elif filter_name == 'brightness':
                brightness = float(filter_params.get('brightness', 0.0))
                filters.append(BrightnessFilter(brightness))
            elif filter_name == 'contrast':
                contrast = float(filter_params.get('contrast', 0.0))
                filters.append(ContrastFilter(contrast))
            elif filter_name == 'saturation':
                saturation = float(filter_params.get('saturation', 0.0))
                filters.append(SaturationFilter(saturation))
            elif filter_name == 'grayscale':
                filters.append(GrayscaleFilter())
            elif filter_name == 'invert':
                filters.append(InvertFilter())
            else:
                raise ValueError(f"Unknown filter: {filter_name}")

        except ValueError as e:
            raise ValueError(f"Error parsing filter '{filter_arg}': {e}")

    return filters
