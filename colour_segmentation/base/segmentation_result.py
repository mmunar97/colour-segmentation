import numpy


class SegmentationResult:
    """
    Object that stores the results of the segmentation algorithm.
    """

    def __init__(self,
                 segmented_image: numpy.ndarray,
                 segmented_classes: numpy.ndarray,
                 elapsed_time: float):
        self.segmented_image = segmented_image
        self.segmented_classes = segmented_classes
        self.elapsed_time = elapsed_time

    def get_colour_proportion(self, colour_label=None):
        """
        Computes the proportion of red pixels in the segmentation.

        Args:
            colour_label: An integer, representing the label associated to the red colour.

        Returns:
            A float, representing the proportion of redness.
        """
        if colour_label is None:
            colour_label = 0

        colour_pixels = self.segmented_classes == colour_label
        return colour_pixels.sum() / (colour_pixels.shape[0] * colour_pixels.shape[1])