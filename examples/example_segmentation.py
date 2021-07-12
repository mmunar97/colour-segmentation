import cv2

from colour_segmentation.base.segmentation_algorithm import SegmentationAlgorithm
from colour_segmentation.segmentator import Segmentator

if __name__ == "__main__":

    image = cv2.imread(r"assets/tree_image.jpg")

    segmentator = Segmentator(image=image)

    result_amante_fonseca = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_AMANTE)
    result_chamorro = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_CHAMORRO)
    result_liu = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_LIU)
    result_shamir = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_SHAMIR)
