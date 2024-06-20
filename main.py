import argparse
from PIL import Image, UnidentifiedImageError
import numpy as np
from cli import parse_args, parse_filters
from processor.image_processor import ImageProcessor

def main():
    try:
        args = parse_args()

        if not args.output and not args.display:
            raise ValueError("At least one of --output or --display must be specified.")

        try:
            # Load image
            image = Image.open(args.input)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{args.input}' not found.")
        except UnidentifiedImageError:
            raise ValueError(f"Input file '{args.input}' is not a valid image.")

        image_array = np.array(image)

        # Create an image processor
        processor = ImageProcessor()

        # Parse and add filters
        filters = parse_filters(args.filters)
        for filter in filters:
            processor.add_filter(filter)

        # Process the image
        processed_image_array = processor.process_image(image_array)

        # Convert back to PIL image
        processed_image = Image.fromarray(processed_image_array)

        # Save or display the result
        if args.output:
            processed_image.save(args.output)
        if args.display:
            processed_image.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
