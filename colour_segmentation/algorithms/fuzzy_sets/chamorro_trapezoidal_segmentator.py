import numpy

import time
from colour_segmentation.base.algorithms.fuzzy_set_segmentator import FuzzySetSegmentator
from skimage import color

from colour_segmentation.base.segmentation_result import SegmentationResult


class ChamorroTrapezoidalSegmentator(FuzzySetSegmentator):

    def __init__(self, image: numpy.ndarray):
        """
        Initializes the object that segments a given image with the Amante-Fonseca fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be inpainted which entries are in 0...255
                   range and the channels are BGR.
        """
        super(ChamorroTrapezoidalSegmentator, self).__init__(image=image,
                                                             class_representation={
                                                                 0: numpy.array([255, 33, 36]),
                                                                 1: numpy.array([255, 148, 9]),
                                                                 2: numpy.array([255, 255, 13]),
                                                                 3: numpy.array([186, 255, 15]),
                                                                 4: numpy.array([6, 155, 9]),
                                                                 5: numpy.array([12, 255, 116]),
                                                                 6: numpy.array([11, 254, 255]),
                                                                 7: numpy.array([8, 192, 255]),
                                                                 8: numpy.array([0, 0, 255]),
                                                                 9: numpy.array([92, 8, 253]),
                                                                 10: numpy.array([238, 3, 249]),
                                                                 11: numpy.array([254, 6, 180])
                                                             })

    def segment(self):
        """
        Segments the image using the Chamorro et al membership functions of different fuzzy sets.

        References:
            Chamorro-Martínez J, Medina JM, Barranco C, Galán-Perales E, Soto-Hidalgo J. (2007)
            Retrieving images in fuzzy object-relational databases using dominant color descriptors
            Fuzzy Sets Systems. 2007 Feb 1; 158:312–24.

        Returns:
            A SegmentationResult object, containing the classification of each pixel and the elapsed time.
        """
        elapsed_time = time.time()

        hsv_image = color.rgb2hsv(self.get_float_image())
        h_channel = 360 * hsv_image[:, :, 0]

        red_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_red)(h_channel)
        orange_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_orange)(h_channel)
        yellow_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_yellow)(h_channel)
        yellowgreen_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_yellowgreen)(h_channel)
        green_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_green)(h_channel)
        greencyan_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_greencyan)(h_channel)
        cyan_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_cyan)(h_channel)
        cyanblue_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_cyanblue)(h_channel)
        blue_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_blue)(h_channel)
        bluemagenta_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_bluemagenta)(h_channel)
        magenta_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_magenta)(h_channel)
        magentared_membership = numpy.vectorize(ChamorroTrapezoidalSegmentator.__fuzzy_trapezoidal_magentared)(h_channel)

        memberships = numpy.stack([red_membership, orange_membership, yellow_membership, yellowgreen_membership,
                                   green_membership, greencyan_membership, cyan_membership, cyanblue_membership,
                                   blue_membership, bluemagenta_membership, magenta_membership,
                                   magentared_membership], axis=2)
        segmentation = self.draw_class_segmentation(classification=memberships.argmax(axis=2))

        return SegmentationResult(segmented_image=segmentation, elapsed_time=time.time() - elapsed_time)

    @staticmethod
    def __fuzzy_trapezoidal_red(h: float):
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
            return -h / 10 + 2
        elif 340 <= h <= 350:
            return h / 10 - 34
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_orange(h: float):
        """
        Computes the membership function of the hue value of the orange fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 10 < h <= 20:
            return h / 10 - 1
        elif 20 <= h <= 40:
            return 1
        elif 40 <= h <= 50:
            return -h / 10 + 5
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_yellow(h: float):
        """
        Computes the membership function of the hue value of the yellow fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 40 <= h <= 50:
            return h / 10 - 4
        elif 50 <= h <= 70:
            return 1
        elif 70 <= h <= 80:
            return -h / 10 + 8
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_yellowgreen(h: float):
        """
        Computes the membership function of the hue value of the yellow-green fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 70 <= h <= 80:
            return h / 10 - 7
        elif 80 <= h <= 100:
            return 1
        elif 100 <= h <= 110:
            return -h / 10 + 11
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_green(h: float):
        """
        Computes the membership function of the hue value of the green fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 100 <= h <= 110:
            return h / 10 - 10
        elif 110 <= h <= 130:
            return 1
        elif 130 <= h <= 140:
            return -h / 10 + 14
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_greencyan(h: float):
        """
        Computes the membership function of the hue value of the green-cyan fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 130 <= h <= 140:
            return h / 10 - 13
        elif 140 <= h <= 160:
            return 1
        elif 160 <= h <= 170:
            return -h / 10 + 17
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_cyan(h: float):
        """
        Computes the membership function of the hue value of the cyan fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 160 <= h <= 170:
            return h / 10 - 16
        elif 170 <= h <= 190:
            return 1
        elif 190 <= h <= 200:
            return -h / 10 + 20
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_cyanblue(h: float):
        """
        Computes the membership function of the hue value of the cyan-blue fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 190 <= h <= 200:
            return h / 10 - 19
        elif 200 <= h <= 220:
            return 1
        elif 220 <= h <= 230:
            return -h / 10 + 23
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_blue(h: float):
        """
        Computes the membership function of the hue value of the blue fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 220 <= h <= 230:
            return h / 10 - 22
        elif 230 <= h <= 250:
            return 1
        elif 250 <= h <= 260:
            return -h / 10 + 26
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_bluemagenta(h: float):
        """
        Computes the membership function of the hue value of the blue-magenta fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 250 <= h <= 260:
            return h / 10 - 25
        elif 260 <= h <= 280:
            return 1
        elif 280 <= h <= 290:
            return -h / 10 + 29
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_magenta(h: float):
        """
        Computes the membership function of the hue value of the magenta fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 280 <= h <= 290:
            return h / 10 - 28
        elif 290 <= h <= 310:
            return 1
        elif 310 <= h <= 320:
            return -h / 10 + 32
        else:
            return 0

    @staticmethod
    def __fuzzy_trapezoidal_magentared(h: float):
        """
        Computes the membership function of the hue value of the magenta-red fuzzy set.

        Args:
            h: The hue value.

        Returns:
            A float, representing the value of the membership function.
        """
        if 310 <= h <= 320:
            return h / 10 - 31
        elif 320 <= h <= 340:
            return 1
        elif 340 <= h <= 350:
            return -h / 10 + 35
        else:
            return 0
