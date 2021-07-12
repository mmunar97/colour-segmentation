import numpy
import time
from skimage import color

from colour_segmentation.base.algorithms.fuzzy_set_segmentator import FuzzySetSegmentator
from colour_segmentation.base.segmentation_result import SegmentationResult


class AmanteTrapezoidalSegmentator(FuzzySetSegmentator):

    def __init__(self, image: numpy.ndarray):
        """
        Initializes the object that segments a given image with the Amante-Fonseca fuzzy sets.

        Args:
            image: A three-dimensional numpy array, representing the image to be segmented which entries are in 0...255
                   range and the channels are BGR.
        """
        super(AmanteTrapezoidalSegmentator, self).__init__(image=image,
                                                           class_representation={
                                                               0: numpy.array([255, 33, 36]),
                                                               1: numpy.array([170, 121, 66]),
                                                               2: numpy.array([255, 146, 0]),
                                                               3: numpy.array([255, 251, 0]),
                                                               4: numpy.array([0, 255, 0]),
                                                               5: numpy.array([0, 253, 255]),
                                                               6: numpy.array([0, 0, 255]),
                                                               7: numpy.array([147, 33, 146]),
                                                               8: numpy.array([255, 64, 255])
                                                           })

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

        hsv_image = color.rgb2hsv(self.get_float_image())
        h_channel = 360*hsv_image[:, :, 0]
        s_channel = hsv_image[:, :, 1]
        v_channel = hsv_image[:, :, 2]

        red_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_red)(h_channel)
        brown_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_brown)(h_channel)
        orange_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_orange)(h_channel)
        yellow_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_yellow)(h_channel)
        green_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_green)(h_channel)
        cyan_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_cyan)(h_channel)
        blue_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_blue)(h_channel)
        purple_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_purple)(h_channel)
        pink_membership = numpy.vectorize(AmanteTrapezoidalSegmentator.__fuzzy_trapezoidal_pink)(h_channel)

        memberships = numpy.stack([red_membership, brown_membership, orange_membership, yellow_membership,
                                   green_membership, cyan_membership, blue_membership,
                                   purple_membership, pink_membership], axis=2)

        segmentation = self.draw_class_segmentation(classification=memberships.argmax(axis=2))

        if remove_achromatic_colours:
            black_pixels = self.__get_black_pixels(v_channel)
            gray_pixels = self.__get_gray_pixels(s_channel, v_channel)
            white_pixels = self.__get_white_pixels(s_channel, v_channel)

            segmentation = AmanteTrapezoidalSegmentator.__graw_achromatic_classes(chromatic_segmentation=segmentation,
                                                                                  black_pixels=black_pixels,
                                                                                  gray_pixels=gray_pixels,
                                                                                  white_pixels=white_pixels)

        elapsed_time = elapsed_time-time.time()

        return SegmentationResult(segmented_image=segmentation,
                                  elapsed_time=elapsed_time,
                                  red_proportion=self.get_red_proportion(segmentation))

    @staticmethod
    def __graw_achromatic_classes(chromatic_segmentation: numpy.ndarray, black_pixels: numpy.ndarray,
                                  gray_pixels: numpy.ndarray, white_pixels: numpy.ndarray):
        """
        Draws over the segmented image the achromatic colours.

        Args:
            chromatic_segmentation: A three-dimensional numpy array, representing the chromatic segmentation.
            black_pixels: A numpy array of booleans, representing the position of the pixels marked as black.
            gray_pixels: A numpy array of booleans, representing the position of the pixels marked as gray.
            white_pixels: A numpy array of booleans, representing the position of the pixels marked as white.

        Returns:
            A three-dimensional numpy array, representing the chromatic segmentation including the achromatic colours.
        """
        chromatic_segmentation[black_pixels, :] = [0, 0, 0]
        chromatic_segmentation[white_pixels, :] = [255, 255, 255]
        chromatic_segmentation[gray_pixels, :] = [128, 128, 128]

        return chromatic_segmentation

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
            return 0

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
            return 0

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
            return 0

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
            return 0

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
            return 0

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
            return 0

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
            return 0

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
            return 0

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
            return 0

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