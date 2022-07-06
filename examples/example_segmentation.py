import cv2
import numpy

from colour_segmentation.base.segmentation_algorithm import SegmentationAlgorithm
from colour_segmentation.segmentator import Segmentator


def load_trial_image(image_name: str) -> numpy.ndarray:
    return cv2.imread(rf"assets/{image_name}")


if __name__ == "__main__":
    # image = load_trial_image("tree.jpg")
    image = load_trial_image("nectarine.jpg")

    segmentator = Segmentator(image=image)

    result_amante_fonseca_chr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_AMANTE,
                                                    remove_achromatic_colours=True)
    result_amante_fonseca_achr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_AMANTE,
                                                     remove_achromatic_colours=False)

    result_liu = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_LIU,
                                     apply_colour_correction=False,
                                     remove_achromatic_colours=True)
    result_liu_corrected = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_LIU,
                                               apply_colour_correction=True,
                                               remove_achromatic_colours=True)

    result_chamorro_chr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_CHAMORRO,
                                              remove_achromatic_colours=True)
    result_chamorro_achr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_CHAMORRO,
                                               remove_achromatic_colours=False)

    result_shamir_chr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_SHAMIR,
                                            remove_achromatic_colours=True)
    result_shamir_achr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_SHAMIR,
                                             remove_achromatic_colours=False)

