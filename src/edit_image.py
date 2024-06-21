import argparse
from PIL import Image, UnidentifiedImageError
import numpy as np
from image_processor import ImageProcessor
from adjustments.brightness_filter import BrightnessFilter
from adjustments.contrast_filter import ContrastFilter
from adjustments.saturation_filter import SaturationFilter
from adjustments.grayscale_filter import GrayscaleFilter
from adjustments.invert_filter import InvertFilter
from filters.box_blur import BoxBlur
from filters.sobel_filter import SobelFilter
from filters.sharpen_filter import SharpenFilter

"""
GEN: Used ChatGPT for initial project structure advice and command-line argument parsing setup.
Prompt: "How to structure a command-line image processing tool with multiple filters in Python? 
I need to support commands like: 
edit-image --input image.png --output result.png --brightness 20 --contrast -3 --display"
"""

def parse_args():
    """
    GEN: ChatGPT assisted in refining argument parser structure.
    Prompt: "How to create a flexible argument parser for multiple image filters in Python?
    I need to support arguments like --brightness, --contrast, --saturation, and --box with values.
    For example: --brightness 20 --box 5 3"
    """
    parser = argparse.ArgumentParser(description='Apply filters to an image.')
    parser.add_argument('--input', required=True, help='Path to the input image')
    parser.add_argument('--output', required=True, help='Path to save the output image')
    parser.add_argument('--display', action='store_true', help='Display the output image')
    parser.add_argument('--brightness', type=float, help='Adjust brightness by specified amount')
    parser.add_argument('--contrast', type=float, help='Adjust contrast by specified amount')
    parser.add_argument('--saturation', type=float, help='Adjust saturation by specified amount')
    parser.add_argument('--grayscale', action='store_true', help='Convert image to grayscale')
    parser.add_argument('--invert', action='store_true', help='Invert image colors')
    parser.add_argument('--box', type=str, help='Apply box blur with specified width and height (e.g., width=5,height=3)')
    parser.add_argument('--sobel', action='store_true', help='Apply Sobel edge detection')
    parser.add_argument('--sharpen', type=float, help='Apply sharpening with specified magnitude')
    
    return parser.parse_args()

def parse_box_params(box_param):
    params = {}
    for param in box_param.split(','):
        key, value = param.split('=')
        params[key] = int(value)
    return params

def main():
    try:
        args = parse_args()
        print("Arguments parsed:", args)

        if not args.output and not args.display:
            raise ValueError("At least one of --output or --display must be specified.")

        try:
            print("Loading image from:", args.input)
            image = Image.open(args.input)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{args.input}' not found.")
        except UnidentifiedImageError:
            raise ValueError(f"Input file '{args.input}' is not a valid image.")

        image_array = np.array(image)
        print("Image loaded successfully")

        processor = ImageProcessor()
        print("Image processor created")

        if args.brightness is not None:
            processor.add_filter(BrightnessFilter(args.brightness))
            print(f"Added BrightnessFilter: {args.brightness}")
        if args.contrast is not None:
            processor.add_filter(ContrastFilter(args.contrast))
            print(f"Added ContrastFilter: {args.contrast}")
        if args.saturation is not None:
            processor.add_filter(SaturationFilter(args.saturation))
            print(f"Added SaturationFilter: {args.saturation}")
        if args.grayscale:
            processor.add_filter(GrayscaleFilter())
            print("Added GrayscaleFilter")
        if args.invert:
            processor.add_filter(InvertFilter())
            print("Added InvertFilter")
        if args.box:
            box_params = parse_box_params(args.box)
            processor.add_filter(BoxBlur(box_params['width'], box_params['height']))
            print(f"Added BoxBlur: {box_params}")
        if args.sobel:
            processor.add_filter(SobelFilter())
            print("Added SobelFilter")
        if args.sharpen is not None:
            processor.add_filter(SharpenFilter(args.sharpen))
            print(f"Added SharpenFilter: {args.sharpen}")

        processed_image_array = processor.process_image(image_array)
        print("Image processed")

        processed_image = Image.fromarray(processed_image_array)

        if args.output:
            processed_image.save(args.output)
            print(f"Image saved to: {args.output}")
        if args.display:
            processed_image.show()
            print("Image displayed")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
