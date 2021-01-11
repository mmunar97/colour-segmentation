import cv2

from colour_segmentation.base.segmentation_algorithm import SegmentationAlgorithm
from colour_segmentation.segmentator import Segmentator

if __name__ == "__main__":

    image = cv2.imread(r"C:\Users\Usuario\Desktop\InpaintingDemo\mumford_shah_clean.png")

    segmentator = Segmentator(image=image)

    result_amante_fonseca = segmentator.segment(method=SegmentationAlgorithm.FUZZY_SET_AMANTE)
