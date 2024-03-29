import cv2
import numpy
import time

from colour_segmentation.base.algorithms.fuzzy_set_segmentator import FuzzySetSegmentator
from colour_segmentation.base.exceptions.fuzzy_palette_invalid_representation import FuzzyPaletteInvalidRepresentation
from colour_segmentation.base.segmentation_result import SegmentationResult
from typing import Dict


class AmanteTrapezoidalSegmentator(FuzzySetSegmentator):

    def __init__(self, image: numpy.ndarray, labels_representation: Dict = None):
        """
        Initializes the object that segments a given image with the Amante-Fonseca fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be segmented which entries are in 0...255
                   range and the channels are BGR.
            labels_representation: A dictionary, representing the palette of colours associated to the Amante-Fonseca
                                   fuzzy sets.
        """
        if not labels_representation:
            labels_representation = {0: numpy.array([255, 33, 36]),
                                     1: numpy.array([170, 121, 66]),
                                     2: numpy.array([255, 146, 0]),
                                     3: numpy.array([255, 251, 0]),
                                     4: numpy.array([0, 255, 0]),
                                     5: numpy.array([0, 253, 255]),
                                     6: numpy.array([0, 0, 255]),
                                     7: numpy.array([147, 33, 146]),
                                     8: numpy.array([255, 64, 255])}

        if len(labels_representation.keys()) != 9:
            raise FuzzyPaletteInvalidRepresentation(provided_labels=len(labels_representation.keys()), needed_labels=9)

        super(AmanteTrapezoidalSegmentator, self).__init__(image=image,
                                                           class_representation=labels_representation)

    def segment(self, remove_achromatic_colours: bool = True) -> SegmentationResult:
        """
        Segments the image using the Amante-Fonseca's membership functions of different fuzzy sets. The method considers
        nine chromatic colour classes, and three achromatic colour classes. Depending on certain conditions, the colours
        present in the image are achromatic colours (black, grey and white).

        Args:
            remove_achromatic_colours: A boolean, indicating if the achromatic colours have to be removed in the image.

        References:
            Amante JC & Fonseca MJ (2012)
            Fuzzy Color Space Segmentation to Identify the Same Dominant Colors as Users.
            18th International Conference on Distributed Multimedia Systems.

        Returns:
            A SegmentationResult object, containing the classification of each pixel and the elapsed time.
        """
        elapsed_time = time.time()

        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h_channel = 2 * (hsv_image[:, :, 0].astype(float))

        red_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_red,
                                         otypes=[numpy.float])(h_channel)
        brown_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_brown,
                                           otypes=[numpy.float])(h_channel)
        orange_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_orange,
                                            otypes=[numpy.float])(h_channel)
        yellow_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_yellow,
                                            otypes=[numpy.float])(h_channel)
        green_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_green,
                                           otypes=[numpy.float])(h_channel)
        cyan_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_cyan,
                                          otypes=[numpy.float])(h_channel)
        blue_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_blue,
                                          otypes=[numpy.float])(h_channel)
        purple_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_purple,
                                            otypes=[numpy.float])(h_channel)
        pink_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_pink,
                                          otypes=[numpy.float])(h_channel)

        memberships = numpy.stack([red_membership, brown_membership, orange_membership, yellow_membership,
                                   green_membership, cyan_membership, blue_membership,
                                   purple_membership, pink_membership], axis=2)
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
    def __fuzzy_trapezoidal_red(h: float) -> float:
        """
        Computes the membership function of the hue value of the red fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 < h <= 10 or 350 < h <= 360:
            return 1
        elif 10 <= h <= 20:
            return 2 - h / 10
        elif 335 <= h <= 350:
            return -335 / 15 + h / 15
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_brown(h: float) -> float:
        """
        Computes the membership function of the hue value of the brown fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 10 < h <= 20:
            return h / 10 - 1
        elif 20 <= h <= 30:
            return 1
        elif 30 <= h <= 35:
            return -h / 5 + 7
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_orange(h: float) -> float:
        """
        Computes the membership function of the hue value of the orange fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 30 <= h <= 34:
            return h / 4 - 15 / 2
        elif 34 <= h <= 42:
            return 1
        elif 42 <= h <= 50:
            return -0.13 * h + 6.25
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_yellow(h: float) -> float:
        """
        Computes the membership function of the hue value of the yellow fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 44 <= h <= 50:
            return 0.17 * h - 7.33
        elif 50 <= h <= 70:
            return 1
        elif 70 <= h <= 100:
            return -0.03 * h + 3.33
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_green(h: float) -> float:
        """
        Computes the membership function of the hue value of the green fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 70 <= h <= 100:
            return 0.03 * h - 2.33
        elif 100 <= h <= 140:
            return 1
        elif 140 <= h <= 160:
            return -0.05 * h + 8
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_cyan(h: float) -> float:
        """
        Computes the membership function of the hue value of the cyan fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 140 <= h <= 160:
            return 0.05 * h - 7
        elif 160 <= h <= 200:
            return 1
        elif 200 <= h <= 220:
            return -0.05 * h + 11
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_blue(h: float) -> float:
        """
        Computes the membership function of the hue value of the blue fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 200 <= h <= 220:
            return 0.05 * h - 10
        elif 220 <= h <= 260:
            return 1
        elif 260 <= h <= 290:
            return -0.03 * h + 9.67
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_purple(h: float) -> float:
        """
        Computes the membership function of the hue value of the purple fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 260 <= h <= 290:
            return 0.03 * h - 8.67
        elif 290 <= h <= 310:
            return 1
        elif 310 <= h <= 320:
            return -h / 10 + 32
        else:
            return 0.0

    @staticmethod
    def __fuzzy_trapezoidal_pink(h: float) -> float:
        """
        Computes the membership function of the hue value of the pink fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 310 <= h <= 315:
            return h / 5 - 62
        elif 315 <= h <= 335:
            return 1
        elif 335 <= h <= 350:
            return -0.07 * h + 23.33
        else:
            return 0.0
