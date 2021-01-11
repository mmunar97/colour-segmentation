import numpy

from colour_segmentation.algorithms.fuzzy_sets.amante_trapezoidal_segmentator import AmanteTrapezoidalSegmentator
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

    def __segment_with_amante_trapezoidal(self) -> SegmentationResult:
        """
        Segments the image with the Amante-Fonseca fuzzy sets.
        """
        fuzzy_set_amante_segmentator = AmanteTrapezoidalSegmentator(image=self.__image)
        return fuzzy_set_amante_segmentator.segment()