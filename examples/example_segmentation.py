import cv2

from colour_segmentation.base.segmentation_algorithm import SegmentationAlgorithm
from colour_segmentation.segmentator import Segmentator

if __name__ == "__main__":
    # image = cv2.imread(r"assets/tree.jpg")
    image = cv2.imread(r"assets/nectarine.jpg")

    segmentator = Segmentator(image=image)

    result_liu = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_LIU,
                                     apply_colour_correction=False,
                                     remove_achromatic_colours=True)
    result_liu_corrected = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_LIU,
                                               apply_colour_correction=True,
                                               remove_achromatic_colours=True)

    result_amante_fonseca_chr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_AMANTE,
                                                    remove_achromatic_colours=True)
    result_amante_fonseca_achr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_AMANTE,
                                                     remove_achromatic_colours=False)

    result_chamorro_chr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_CHAMORRO,
                                              remove_achromatic_colours=True)
    result_chamorro_achr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_CHAMORRO,
                                               remove_achromatic_colours=False)

    result_shamir_chr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_SHAMIR,
                                            remove_achromatic_colours=True)
    result_shamir_achr = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_SHAMIR,
                                             remove_achromatic_colours=False)
