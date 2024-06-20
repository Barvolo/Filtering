import argparse
from PIL import Image
import numpy as np
from cli.parser import parse_args, parse_filters
from processor.image_processor import ImageProcessor

def main():
    args = parse_args()
    if not args.output and not args.display:
        raise ValueError("At least one of --output or --display must be specified.")

    # Load image
    image = Image.open(args.input)
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

if __name__ == "__main__":
    main()
