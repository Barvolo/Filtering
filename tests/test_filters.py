import unittest
import numpy as np
from filters.box_blur import BoxBlur
from filters.sobel_filter import SobelFilter
from filters.sharpen_filter import SharpenFilter
from processor.image_processor import ImageProcessor
import pdb


class TestBoxBlur(unittest.TestCase):
    def setUp(self):
        # Create simple test images
        self.test_image_3x3 = np.array([[100, 150, 100],
                                        [150, 200, 150],
                                        [100, 150, 100]], dtype=np.uint8)
        
        self.test_image_5x5 = np.array([[100, 150, 200, 150, 100],
                                        [150, 200, 250, 200, 150],
                                        [200, 250, 255, 250, 200],
                                        [150, 200, 250, 200, 150],
                                        [100, 150, 200, 150, 100]], dtype=np.uint8)
        
        self.test_image_7x7 = np.array([[100, 150, 200, 250, 200, 150, 100],
                                        [150, 200, 250, 255, 250, 200, 150],
                                        [200, 250, 255, 255, 255, 250, 200],
                                        [250, 255, 255, 255, 255, 255, 250],
                                        [200, 250, 255, 255, 255, 250, 200],
                                        [150, 200, 250, 255, 250, 200, 150],
                                        [100, 150, 200, 250, 200, 150, 100]], dtype=np.uint8)

    def test_box_blur_3x3(self):
        blur = BoxBlur(width=3, height=3)
        output = blur.apply(self.test_image_3x3)
        expected_output = np.array([
            [150, 141, 150],
            [141, 133, 141],
            [150, 141, 150]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected_output, err_msg="Box Blur 3x3 failed")

    def test_box_blur_5x5(self):
        blur = BoxBlur(width=3, height=3)
        output = blur.apply(self.test_image_5x5)
        expected_output = np.array([
            [150, 175, 191, 175, 150],
            [175, 195, 211, 195, 175],
            [191, 211, 228, 211, 191],
            [175, 195, 211, 195, 175],
            [150, 175, 191, 175, 150]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected_output, err_msg="Box Blur 5x5 failed")

    def test_box_blur_7x7(self):
        blur = BoxBlur(width=3, height=3)
        output = blur.apply(self.test_image_7x7)
        expected_output = np.array([
            [150, 175, 217, 234, 217, 175, 150],
            [175, 195, 229, 241, 229, 195, 175],
            [217, 229, 247, 253, 247, 229, 217],
            [234, 241, 253, 255, 253, 241, 234],
            [217, 229, 247, 253, 247, 229, 217],
            [175, 195, 229, 241, 229, 195, 175],
            [150, 175, 217, 234, 217, 175, 150]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected_output, err_msg="Box Blur 7x7 failed")


class TestSobelFilter(unittest.TestCase):
    def setUp(self):
        self.test_image_3x3 = np.array([
            [100, 150, 100],
            [150, 200, 150],
            [100, 150, 100]
        ], dtype=np.uint8)
        
        self.test_image_5x5 = np.array([
            [100, 150, 200, 150, 100],
            [150, 200, 250, 200, 150],
            [200, 250, 255, 250, 200],
            [150, 200, 250, 200, 150],
            [100, 150, 200, 150, 100]
        ], dtype=np.uint8)
        
        self.test_image_7x7 = np.array([
            [100, 150, 200, 250, 200, 150, 100],
            [150, 200, 250, 255, 250, 200, 150],
            [200, 250, 255, 255, 255, 250, 200],
            [250, 255, 255, 255, 255, 255, 250],
            [200, 250, 255, 255, 255, 250, 200],
            [150, 200, 250, 255, 250, 200, 150],
            [100, 150, 200, 250, 200, 150, 100]
        ], dtype=np.uint8)

    def test_sobel_3x3(self):
        sobel = SobelFilter()
        output = sobel.apply(self.test_image_3x3)
        expected_output = np.array([
            [255, 255, 255],
            [255,   0, 255],
            [255, 255, 255]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_output, err_msg="Sobel 3x3 failed")

    def test_sobel_5x5(self):
        sobel = SobelFilter()
        output = sobel.apply(self.test_image_5x5)
        expected_output = np.array([
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
            [255, 255,   0, 255, 255],
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_output, err_msg="Sobel 5x5 failed")

    def test_sobel_7x7(self):
        sobel = SobelFilter()
        output = sobel.apply(self.test_image_7x7)
        expected_output = np.array([
            [255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 120, 255, 255, 255],
            [255, 255,  91,  10,  91, 255, 255],
            [255, 120,  10,   0,  10, 120, 255],
            [255, 255,  91,  10,  91, 255, 255],
            [255, 255, 255, 120, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_output, err_msg="Sobel 7x7 failed")


class TestSharpenFilter(unittest.TestCase):
    def setUp(self):
        # Create a 3x3 test image
        self.test_image_3x3 = np.array([
            [10, 10, 10],
            [10, 50, 10],
            [10, 10, 10]
        ], dtype=np.uint8)
        
        # Create a 5x5 test image
        self.test_image_5x5 = np.array([
            [10, 10, 10, 10, 10],
            [10, 50, 50, 50, 10],
            [10, 50, 100, 50, 10],
            [10, 50, 50, 50, 10],
            [10, 10, 10, 10, 10]
        ], dtype=np.uint8)

    def test_sharpen_3x3(self):
        sharpen = SharpenFilter(magnitude=0.5)
        
        # Apply Sharpen filter
        output = sharpen.apply(self.test_image_3x3)
        
        # Expected edge detection result for 3x3 image
        expected_output = np.array([
            [59, 70, 59],
            [70, 50, 70],
            [59, 70, 59]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(output, expected_output, err_msg="Sharpen 3x3 failed")

    def test_sharpen_5x5(self):
        sharpen = SharpenFilter(magnitude=0.5)
        
        # Apply Sharpen filter
        output = sharpen.apply(self.test_image_5x5)
    

        # Expected edge detection result for 5x5 image
        expected_output = np.array([
            [59, 92, 110, 92, 59],
            [92, 170, 177, 170, 92],
            [110,177 ,100 ,177 ,110],
            [92, 170, 177, 170, 92],
            [59, 92, 110, 92, 59]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(output, expected_output, err_msg="Sharpen 5x5 failed")


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.test_image_3x3 = np.array([
            [10, 10, 10],
            [10, 50, 10],
            [10, 10, 10]
        ], dtype=np.uint8)

        self.test_image_5x5 = np.array([
            [10, 10, 10, 10, 10],
            [10, 50, 50, 50, 10],
            [10, 50, 100, 50, 10],
            [10, 50, 50, 50, 10],
            [10, 10, 10, 10, 10]
        ], dtype=np.uint8)

    def test_image_processor_with_multiple_filters(self):
        processor = ImageProcessor()
        
        box_blur = BoxBlur(width=3, height=3)
        sobel_filter = SobelFilter()
        sharpen_filter = SharpenFilter(magnitude=0.5)

        processor.add_filter(box_blur)
        
        processor.add_filter(sharpen_filter)
        processor.add_filter(sobel_filter)

        processed_image = self.test_image_3x3
        for filter in processor.filters:
            pdb.set_trace()
            processed_image = filter.apply(processed_image)
            print(f"After {filter.__class__.__name__}: \n{processed_image}\n")

        processed_image = self.test_image_5x5
        for filter in processor.filters:
            pdb.set_trace()
            processed_image = filter.apply(processed_image)
            print(f"After {filter.__class__.__name__}: \n{processed_image}\n")

if __name__ == '__main__':
    unittest.main()
