
# Advanced Image Editing CLI Tool

## Project Overview

Welcome to the Advanced Image Editing CLI Tool! This project focuses on developing advanced solutions for image processing. The CLI tool enables users to apply custom filters and adjustments to images, ensuring a user-friendly and extendable code structure.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Examples](#examples)
7. [Filter Explanations](#filter-explanations)
8. [Development and Testing](#development-and-testing)
9. [Acknowledgements](#acknowledgements)

## Project Structure

\`\`\`
edit-image/
├── src/
│   ├── base/
│   │   ├── __init__.py
│   │   └── base_filter.py
│   ├── adjustments/
│   │   ├── __init__.py
│   │   ├── brightness.py
│   │   ├── contrast.py
│   │   ├── grayscale.py
│   │   ├── invert.py
│   │   └── saturation.py
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── box_blur.py
│   │   ├── sharpen.py
│   │   └── sobel.py
│   ├── edit_image.py
│   ├── image_processor.py
│   └── setup.py
├── tests/
│   ├── __init__.py
│   └── test_filters.py
├── images/
│   ├── input/
│   │   └── example.png
│   └── output/
│       └── processed.png
└── README.md
\`\`\`

## Features

- **Box Blur**: Soften the image by averaging the pixel values in a neighborhood around each pixel.
- **Edge Detection (Sobel)**: Highlight the edges in the image using the Sobel operator.
- **Sharpening**: Enhance image edges by adding edge detection results to the original image.
- **Brightness Adjustment**: Modify the brightness of the image.
- **Contrast Adjustment**: Adjust the contrast of the image.
- **Saturation Adjustment**: Fine-tune the color saturation of the image.
- **Grayscale**: Convert the image to grayscale.
- **Invert Colors**: Invert the colors of the image.

## Dependencies

- Python 3.7+
- Pillow
- NumPy

## Installation

To install the CLI tool, follow these steps:

1. Clone the repository:
   \`\`\`sh
   git clone https://github.com/your-username/edit-image.git
   cd edit-image/src
   \`\`\`

2. Install the dependencies:
   \`\`\`sh
   pip install -r requirements.txt
   \`\`\`

3. Install the CLI tool:
   \`\`\`sh
   pip install -e .
   \`\`\`

## Usage

The command should follow this structure:
\`\`\`sh
edit-image --input <path-to-image> [--<feature-name> <feature-specific-arguments>...]... [--display] [--output <output-path>]
\`\`\`

Example:
\`\`\`sh
edit-image --input images/input/example.png --brightness 20 --contrast -3 --box width=5,height=5 --output images/output/processed.png --display
\`\`\`

### Available Features

- \`--brightness <value>\`: Adjust brightness (e.g., --brightness 20)
- \`--contrast <value>\`: Adjust contrast (e.g., --contrast -3)
- \`--saturation <value>\`: Adjust saturation (e.g., --saturation 1.5)
- \`--grayscale\`: Convert to grayscale
- \`--invert\`: Invert colors
- \`--box width=<value>,height=<value>\`: Apply box blur (e.g., --box width=5,height=5)
- \`--sobel\`: Apply Sobel edge detection
- \`--sharpen <value>\`: Sharpen image with magnitude (e.g., --sharpen 1.5)
- \`--display\`: Display the processed image
- \`--output <path>\`: Save the processed image to the specified path

## Examples

### Example 1: Adjust Brightness and Contrast
\`\`\`sh
edit-image --input images/input/example.png --brightness 20 --contrast -3 --output images/output/brightness_contrast.png --display
\`\`\`
![Example 1 Output](images/output/charizardcharizard.png)

### Example 2: Apply Box Blur and Sharpening
\`\`\`sh
edit-image --input images/input/example.png --box width=5,height=5 --sharpen 1.5 --output images/output/box_sharpen.png --display
\`\`\`
![Example 2 Output](images/output/charizard.jpg)

### Example 3: Convert to Grayscale and apply Sobel and Invert Colors
\`\`\`sh
edit-image --input images/input/example.png --grayscale --sobel --invert --output images/output/grayscale_invert.png --display
\`\`\`
![Example 3 Output](images/output/charizard.jpg)

## Filter Explanations

### Box Blur
The Box Blur filter softens the image by averaging the pixel values within a defined neighborhood around each pixel. This creates a smoothing effect, reducing noise and details.

### Edge Detection (Sobel)
The Sobel operator is used to highlight edges within an image by detecting horizontal and vertical changes. This operator applies two convolution matrices designed to detect these changes, essentially outlining the edges. By default, it works with the base color of the image, but it is recommended to use it with the grayscale filter beforehand for better edge detection results. This is because edge detection is often more effective in grayscale, where variations in intensity are more pronounced.

### Sharpening
The Sharpen filter enhances the edges by adding the results from the edge detection (like the Sobel filter) back to the original image. This increases the contrast of the edges, making the image appear more defined.

## Development and Testing

### Running Tests

1. Navigate to the project root directory:
   \`\`\`sh
   cd edit-image
   \`\`\`

2. Run the tests using \`unittest\`:
   \`\`\`sh
   python -m unittest discover tests
   \`\`\`

### Testing Overview

The tests are designed to check the basic operations of the image filters on small matrices to ensure that the algorithms are implemented correctly. This includes verifying the output of filters like Box Blur, Sobel, and Sharpen on predefined test images.


---
