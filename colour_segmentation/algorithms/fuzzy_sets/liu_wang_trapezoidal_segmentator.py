from skimage import color
import time
import numpy

from colour_segmentation.base.algorithms.fuzzy_set_segmentator import FuzzySetSegmentator
from colour_segmentation.base.segmentation_result import SegmentationResult


class LiuWangTrapezoidalSegmentator(FuzzySetSegmentator):

    def __init__(self, image: numpy.ndarray):
        """
        Initializes the object that segments a given image with the Liu-Wang fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be inpainted which entries are in 0...255
                   range and the channels are BGR.
        """
        super(LiuWangTrapezoidalSegmentator, self).__init__(image=image,
                                                            class_representation={
                                                                0: numpy.array([255, 33, 36]),
                                                                1: numpy.array([248, 149, 29]),
                                                                2: numpy.array([239, 233, 17]),
                                                                3: numpy.array([105, 189, 69]),
                                                                4: numpy.array([111, 204, 221]),
                                                                5: numpy.array([59, 83, 164]),
                                                                6: numpy.array([158, 80, 159])
                                                            })

    def segment(self):
        """
        Segments the image using the liu-Wang membership functions of different fuzzy sets.

        References:
            Liu C, Wang L. (2016)
            Fuzzy color recognition and segmentation of robot vision scene.
            Proceedings - 8th International Congress of Image Signal Processing CISP 2015. 2016;(Cisp):448–52.

        Returns:
            A SegmentationResult object, containing the classification of each pixel and the elapsed time.
        """
        elapsed_time = time.time()

        hsv_image = color.rgb2hsv(self.get_float_image())
        h_channel = 360 * hsv_image[:, :, 0]

        red_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_red)(h_channel)
        orange_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_orange)(h_channel)
        yellow_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_yellow)(h_channel)
        green_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_green)(h_channel)
        cyan_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_cyan)(h_channel)
        blue_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_blue)(h_channel)
        purple_membership = numpy.vectorize(LiuWangTrapezoidalSegmentator.__fuzzy_trapezoidal_purple)(h_channel)

        memberships = numpy.stack([red_membership, orange_membership, yellow_membership, green_membership,
                                   cyan_membership, blue_membership, purple_membership], axis=2)

        segmentation = self.draw_class_segmentation(classification=memberships.argmax(axis=2))

        return SegmentationResult(segmented_image=segmentation, elapsed_time=time.time() - elapsed_time)

    @staticmethod
    def __fuzzy_trapezoidal_red(h: float) -> float:
        """
        Computes the membership function of the hue value of the red fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 10 or 330 < h <= 360:
            return 1
        elif 10 < h <= 20:
            return -0.1 * h + 2
        elif 20 < h <= 300:
            return 0
        elif 300 < h <= 330:
            return h / 30 - 10

    @staticmethod
    def __fuzzy_trapezoidal_orange(h: float) -> float:
        """
        Computes the membership function of the hue value of the orange fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 10 or 55 < h <= 360:
            return 0
        elif 10 < h <= 20:
            return 0.1*h-1
        elif 20 < h <= 40:
            return 1
        elif 40 < h <= 55:
            return -h/15+11/3

    @staticmethod
    def __fuzzy_trapezoidal_yellow(h: float) -> float:
        """
        Computes the membership function of the hue value of the yellow fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 40 or 80 < h <= 360:
            return 0
        elif 40 < h <= 55:
            return h/15-8/3
        elif 55 < h <= 65:
            return 1
        elif 65 < h <= 80:
            return -h/15+11/3

    @staticmethod
    def __fuzzy_trapezoidal_green(h: float) -> float:
        """
        Computes the membership function of the hue value of the green fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 65 or 170 < h <= 360:
            return 0
        elif 65 < h <= 80:
            return h/15-8/3
        elif 80 < h <= 140:
            return 1
        elif 140 < h <= 170:
            return -h/30+17/3

    @staticmethod
    def __fuzzy_trapezoidal_cyan(h: float) -> float:
        """
        Computes the membership function of the hue value of the cyan fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 140 or 210 < h <= 360:
            return 0
        elif 140 < h <= 170:
            return h/30-14/3
        elif 170 < h <= 200:
            return 1
        elif 200 < h <= 210:
            return -h/10+21

    @staticmethod
    def __fuzzy_trapezoidal_blue(h: float) -> float:
        """
        Computes the membership function of the hue value of the blue fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 200 or 270 < h <= 360:
            return 0
        elif 200 <= h <= 210:
            return h/10-20
        elif 210 <= h <= 250:
            return 1
        elif 250 <= h <= 270:
            return -h/20+27/2

    @staticmethod
    def __fuzzy_trapezoidal_purple(h: float) -> float:
        """
        Computes the membership function of the hue value of the purple fuzzy set.

        References:
            1. Liu C, Wang L. Fuzzy color recognition and segmentation of robot vision scene.
            Proc - 2015 8th Int Congr Image Signal Process CISP 2015. 2016;(Cisp):448–52.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 0 <= h <= 250 or 330 < h <= 360:
            return 0
        elif 250 < h <= 270:
            return h/20-5/4
        elif 270 < h <= 300:
            return 1
        elif 300 < h <= 330:
            return -h/30+11