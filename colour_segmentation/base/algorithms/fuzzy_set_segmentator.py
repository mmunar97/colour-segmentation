import numpy

from skimage import img_as_float
from typing import Dict, List

from colour_segmentation.base.segmentation_result import SegmentationResult


class FuzzySetSegmentator:

    def __init__(self, image: numpy.ndarray, class_representation: Dict):
        """
        Initializes the base object for the segmentation using the membership functions of fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be segmented which entries are in 0...255
                   range and the channels are BGR.
            class_representation: A dictionary with the representation colour of each class. Each entry in the dictionary
                                  must be an integer as the key, and a RGB tuple as value.
        """
        self.image = image
        self.class_representation = class_representation

    def segment(self, **kwargs) -> SegmentationResult:
        pass

    def get_float_image(self) -> numpy.ndarray:
        """
        Converts the image from BGR representation and entries in 0...255 range into RGB representation and entries
        in 0...1 range.

        Returns:
            A numpy array, representing the image in the new format.
        """
        return img_as_float(self.image[:, :, ::-1])

    def draw_class_segmentation(self, classification: numpy.ndarray) -> numpy.ndarray:
        """
        Draws the representation color of each class in the corresponding pixel.

        Args:
            classification: A numpy array of integers. Each integer represents the code associated to each class.

        Returns:
            A numpy array, representing the segmented image.
        """
        segmentation = numpy.zeros_like(self.image)
        for class_value in self.class_representation.keys():
            segmentation[classification == class_value] = self.class_representation[class_value]

        return segmentation[:, :, ::-1]

    def get_red_proportion(self, segmentation: numpy.ndarray, red_representation=None):
        """
        Computes the proportion of red pixels in the segmentation.

        Args:
            segmentation: A three-dimensional numpy array, representing the segmented image. Each entry contains the
                          representation colour of the original pixel.
            red_representation: A list of integers, representing the RGB representation of the red colour.

        Returns:
            A float, representing the proportion of redness.
        """
        if red_representation is None:
            red_representation = [255, 33, 36]

        red_pixels = numpy.any(segmentation == red_representation, axis=-1)
        return red_pixels.sum() / (segmentation.shape[0] * segmentation.shape[1])
