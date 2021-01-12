import numpy

from colour_segmentation.algorithms.fuzzy_sets.amante_trapezoidal_segmentator import AmanteTrapezoidalSegmentator
from colour_segmentation.algorithms.fuzzy_sets.chamorro_trapezoidal_segmentator import ChamorroTrapezoidalSegmentator
from colour_segmentation.algorithms.fuzzy_sets.liu_wang_trapezoidal_segmentator import LiuWangTrapezoidalSegmentator
from colour_segmentation.algorithms.fuzzy_sets.shamir_triangular_segmentator import ShamirTriangularSegmentator
from colour_segmentation.base.segmentation_algorithm import SegmentationAlgorithm
from colour_segmentation.base.segmentation_result import SegmentationResult


class Segmentator:

    def __init__(self, image: numpy.ndarray):
        """
        Initializes the object that segments a given image into its main colours.

        Args:
            image: A three-dimensional numpy array, representing the image to be segmented, which entries are in 0...255
                   range and the channels are BGR.
        """
        self.__image = image

    def segment(self, method: SegmentationAlgorithm) -> SegmentationResult:
        """
        Segments the image with a certain method.

        Args:
            method: A SegmentationAlgorithm value, representing the method to be used.

        Returns:
            A SegmentationResult object, containing the classification of each pixel and the elapsed time.
        """
        if method == SegmentationAlgorithm.FUZZY_SET_AMANTE:
            return self.__segment_with_amante_trapezoidal()
        elif method == SegmentationAlgorithm.FUZZY_SET_CHAMORRO:
            return self.__segment_with_chamorro_trapezoidal()
        elif method == SegmentationAlgorithm.FUZZY_SET_LIU:
            return self.__segment_with_liu_trapezoidal()
        elif method == SegmentationAlgorithm.FUZZY_SET_SHAMIR:
            return self.__segment_with_shamir_triangular()

    def __segment_with_amante_trapezoidal(self) -> SegmentationResult:
        """
        Segments the image with the Amante-Fonseca fuzzy sets.
        """
        fuzzy_set_amante_segmentator = AmanteTrapezoidalSegmentator(image=self.__image)
        return fuzzy_set_amante_segmentator.segment()

    def __segment_with_chamorro_trapezoidal(self) -> SegmentationResult:
        """
        Segments the image with the Chamorro et al fuzzy sets.
        """
        fuzzy_set_chamorro_segmentator = ChamorroTrapezoidalSegmentator(image=self.__image)
        return fuzzy_set_chamorro_segmentator.segment()

    def __segment_with_liu_trapezoidal(self) -> SegmentationResult:
        """
        Segments the image with the Liu-Wang fuzzy sets.
        """
        fuzzy_set_liu_segmentator = LiuWangTrapezoidalSegmentator(image=self.__image)
        return fuzzy_set_liu_segmentator.segment()

    def __segment_with_shamir_triangular(self) -> SegmentationResult:
        """
        Segments the image with the Shamir fuzzy sets.
        """
        fuzzy_set_shamir_segmentator = ShamirTriangularSegmentator(image=self.__image)
        return fuzzy_set_shamir_segmentator.segment()
