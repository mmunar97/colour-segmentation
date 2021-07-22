import numpy

from skimage import img_as_float
from typing import Dict

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
        """
        A generic method to compute the colour segmentation of an RGB image.
        """
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

    @staticmethod
    def draw_achromatic_classes(s_channel: numpy.ndarray, v_channel: numpy.ndarray,
                                chromatic_segmentation: numpy.ndarray,
                                colour_classes_segmentation: numpy.ndarray):
        """
        Draws over the segmented image the achromatic colours. The definition of achromatic colours is given in Amante
        et al.

        References:
            Amante JC & Fonseca MJ (2012)
            Fuzzy Color Space Segmentation to Identify the Same Dominant Colors as Users.
            18th International Conference on Distributed Multimedia Systems.

        Args:
            s_channel: A two-dimensional numpy array, representing the saturation channel of the HSV colour space.
            v_channel: A two-dimensional numpy array, representing the intensity channel of the HSV colour space.
            chromatic_segmentation: A three-dimensional numpy array, representing the chromatic segmentation.
            colour_classes_segmentation: A two-dimensional numpy array, representing the class label of the chromatic
                                         segmentation.

        Returns:
            A three-dimensional numpy array, representing the chromatic segmentation including the achromatic colours.
        """
        chr_segm = chromatic_segmentation.copy()
        classes = colour_classes_segmentation.copy()

        black_pixels = FuzzySetSegmentator.__get_black_pixels(v_channel)
        gray_pixels = FuzzySetSegmentator.__get_gray_pixels(s_channel, v_channel)
        white_pixels = FuzzySetSegmentator.__get_white_pixels(s_channel, v_channel)

        chr_segm[black_pixels, :] = [0, 0, 0]
        classes[black_pixels] = -1
        chr_segm[white_pixels, :] = [255, 255, 255]
        classes[white_pixels] = -2
        chr_segm[gray_pixels, :] = [128, 128, 128]
        classes[gray_pixels] = -3

        return chr_segm, classes

    @staticmethod
    def __get_white_pixels(s_channel: numpy.ndarray, v_channel: numpy.ndarray) -> numpy.ndarray:
        """
        Calculates the mask for pixels that are marked as white. To do this, check which pixels have a value of V
        greater than 0.81 and a value of S less than 0.14.

        Args:
            s_channel: A numpy array, representing the matrix of the saturation.
            v_channel: A numpy array, representing the matrix of the values.

        Returns:
            A numpy array of booleans, marking with 1 if the pixel is white, or with 0 in other case.
        """
        return numpy.logical_and(v_channel > 0.81, s_channel <= 0.14)

    @staticmethod
    def __get_black_pixels(v_channel: numpy.ndarray) -> numpy.ndarray:
        """
        Calculates the mask for pixels that are marked as black. To do this, check which pixels have a value of V
        less than 0.19.

        Args:
            v_channel: A numpy array, representing the matrix of the values.

        Returns:
            A numpy array of booleans, marking with 1 if the pixel is black, or with 0 in other case.
        """
        return v_channel <= 0.19

    @staticmethod
    def __get_gray_pixels(s_channel: numpy.ndarray, v_channel: numpy.ndarray) -> numpy.ndarray:
        """
        Calculates the mask for pixels that are marked as gray. To do this, check which pixels have a value of V
        less than 0.81 and greather than 0.19, and a value of S less than 0.14.

        Args:
            s_channel: A numpy array, representing the matrix of the saturation.
            v_channel: A numpy array, representing the matrix of the values.

        Returns:
            A numpy array of booleans, marking with 1 if the pixel is gray, or with 0 in other case.
        """
        v1 = v_channel <= 0.81
        v2 = 0.19 < v_channel

        return numpy.logical_and(numpy.logical_and(v1, v2), s_channel <= 0.14)
