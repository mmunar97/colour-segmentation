import numpy


class SegmentationResult:
    """
    Object that stores the results of the segmentation algorithm.
    """

    def __init__(self, segmented_image: numpy.ndarray,
                 elapsed_time: float):
        self.segmented_image = segmented_image
        self.elapsed_time = elapsed_time
