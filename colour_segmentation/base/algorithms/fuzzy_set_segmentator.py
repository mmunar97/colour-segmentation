import numpy

from skimage import img_as_float
from typing import Dict

from colour_segmentation.base.segmentation_result import SegmentationResult


class FuzzySetSegmentator:

    def __init__(self, image: numpy.ndarray, class_representation: Dict):
        """
        Initializes the base object for the segmentation using the membership functions of fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be inpainted which entries are in 0...255
                   range and the channels are BGR.
            class_representation: A dictionary with the representation colour of each class. Each entry in the dictionary
                                  must be an integer as the key, and a RGB tuple as value.
        """
        self.image = image
        self.class_representation = class_representation

    def segment(self) -> SegmentationResult:
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
