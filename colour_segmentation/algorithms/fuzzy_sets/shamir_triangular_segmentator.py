import cv2
import numpy
import time

from colour_segmentation.base.algorithms.fuzzy_set_segmentator import FuzzySetSegmentator
from colour_segmentation.base.exceptions.FuzzyPaletteInvalidRepresentation import FuzzyPaletteInvalidRepresentation
from colour_segmentation.base.segmentation_result import SegmentationResult
from typing import Dict


class ShamirTriangularSegmentator(FuzzySetSegmentator):

    def __init__(self, image: numpy.ndarray, labels_representation: Dict = None):
        """
        Initializes the object that segments a given image with the Shamir fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be segmented which entries are in 0...255
                   range and the channels are BGR.
        """
        if not labels_representation:
            labels_representation = {0: numpy.array([255, 33, 36]),
                                     1: numpy.array([255, 140, 0]),
                                     2: numpy.array([255, 165, 0]),
                                     3: numpy.array([255, 255, 0]),
                                     4: numpy.array([144, 238, 144]),
                                     5: numpy.array([0, 100, 0]),
                                     6: numpy.array([0, 255, 255]),
                                     7: numpy.array([0, 0, 255]),
                                     8: numpy.array([128, 0, 128]),
                                     9: numpy.array([255, 0, 255])}

        if len(labels_representation.keys()) != 10:
            raise FuzzyPaletteInvalidRepresentation(provided_labels=len(labels_representation.keys()), needed_labels=10)

        super(ShamirTriangularSegmentator, self).__init__(image=image,
                                                          class_representation=labels_representation)

    def segment(self, remove_achromatic_colours: bool = True) -> SegmentationResult:
        """
        Segments the image using the Shamir membership functions of different fuzzy sets.

        Args:
            remove_achromatic_colours: A boolean, indicating if the achromatic colours have to be removed in the image.

        References:
            Shamir, L. (2006)
            Human Perception-based Color Segmentation Using Fuzzy Logic.
            Proceedings of International Conference on Image Processing, Computer Vision, & Pattern Recognition.

        Returns:
            A SegmentationResult object, containing the classification of each pixel and the elapsed time.
        """
        elapsed_time = time.time()

        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h_channel = 2 * (hsv_image[:, :, 0].astype(float))

        red_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_red,
                                         otypes=[numpy.float])(h_channel)
        darkorange_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_darkorange,
                                                otypes=[numpy.float])(h_channel)
        lightorange_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_lightorange,
                                                 otypes=[numpy.float])(h_channel)
        yellow_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_yellow,
                                            otypes=[numpy.float])(h_channel)
        lightgreen_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_lightgreen,
                                                otypes=[numpy.float])(h_channel)
        darkgreen_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_darkgreen,
                                               otypes=[numpy.float])(h_channel)
        aqua_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_aqua,
                                          otypes=[numpy.float])(h_channel)
        blue_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_blue,
                                          otypes=[numpy.float])(h_channel)
        darkpurple_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_darkpurple,
                                                otypes=[numpy.float])(h_channel)
        lightpurple_membership = numpy.vectorize(ShamirTriangularSegmentator.__fuzzy_triangular_lightpurple,
                                                 otypes=[numpy.float])(h_channel)

        memberships = numpy.stack([red_membership, darkorange_membership, lightorange_membership, yellow_membership,
                                   lightgreen_membership, darkgreen_membership, aqua_membership,
                                   blue_membership, darkpurple_membership, lightpurple_membership], axis=2)
        colour_classes = memberships.argmax(axis=2)
        segmentation = self.draw_class_segmentation(classification=colour_classes)

        if remove_achromatic_colours:
            s_channel = (hsv_image[:, :, 1].astype(float)) / 255
            v_channel = (hsv_image[:, :, 2].astype(float)) / 255

            segmentation, colour_classes = self.draw_achromatic_classes(s_channel=s_channel,
                                                                        v_channel=v_channel,
                                                                        chromatic_segmentation=segmentation,
                                                                        colour_classes_segmentation=colour_classes)

        elapsed_time = elapsed_time - time.time()

        return SegmentationResult(segmented_image=segmentation,
                                  segmented_classes=colour_classes,
                                  elapsed_time=elapsed_time)

    @staticmethod
    def __fuzzy_triangular_red(h: float) -> float:
        """
        Computes the membership function of the hue value of the red fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 30:
            return 1 - h / 30
        elif 330 <= h <= 360:
            return -11 + h / 30
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_darkorange(h: float) -> float:
        """
        Computes the membership function of the hue value of the dark orange fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 30:
            return h / 30
        elif 30 <= h <= 45:
            return 3 - h / 15
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_lightorange(h: float) -> float:
        """
        Computes the membership function of the hue value of the light orange fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 30 <= h <= 45:
            return -2 + h / 15
        elif 45 <= h <= 60:
            return 4 - h / 15
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_yellow(h: float) -> float:
        """
        Computes the membership function of the hue value of the yellow fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 45 <= h <= 60:
            return -3 + h / 15
        elif 60 <= h <= 90:
            return 3 - h / 30
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_lightgreen(h: float) -> float:
        """
        Computes the membership function of the hue value of the light green fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 60 <= h <= 75:
            return -4 + h / 15
        elif 75 <= h <= 120:
            return 8 / 3 - h / 45
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_darkgreen(h: float) -> float:
        """
        Computes the membership function of the hue value of the dark green fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 90 <= h <= 120:
            return -3 + h / 30
        elif 120 <= h <= 180:
            return 3 - h / 60
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_aqua(h: float) -> float:
        """
        Computes the membership function of the hue value of the aqua fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 120 <= h <= 180:
            return -2 + h / 60
        elif 180 <= h <= 240:
            return 4 - h / 60
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_blue(h: float) -> float:
        """
        Computes the membership function of the hue value of the blue fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 180 <= h <= 240:
            return -3 + h / 60
        elif 240 <= h <= 300:
            return 5 - h / 60
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_darkpurple(h: float) -> float:
        """
        Computes the membership function of the hue value of the dark purple fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 240 <= h <= 300:
            return -4 + h / 60
        elif 300 <= h <= 330:
            return 11 - h / 30
        else:
            return 0.0

    @staticmethod
    def __fuzzy_triangular_lightpurple(h: float) -> float:
        """
        Computes the membership function of the hue value of the light purple fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 300 <= h <= 330:
            return -10 + h / 30
        elif 330 <= h <= 360:
            return 12 - h / 30
        else:
            return 0.0
