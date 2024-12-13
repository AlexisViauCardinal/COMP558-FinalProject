import numpy as np
import cv2 as cv
from feature_description.feature_descriptor import FeatureDescriptor

class SIFTDescriptor(FeatureDescriptor):

    def __init__(self, params = {}):
        super().__init__()
        self.sift_obj = cv.SIFT.create(**params)

    def detect_features(self, image : np.ndarray, mask : np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        '''
            Computes distinctive SIFT features

            image: an BGR image as an N*M*3 array (np.ndarray)
            mask: an N*M array masking possible point location (1 is to include, 0 is to exclude)

            Returns an (N, 2) array (np.ndarray) containing the points location and associated feature descriptors as an (N, M) array
        '''

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        loc, desc = self.sift_obj.detectAndCompute(gray, mask = mask)

        return cv.KeyPoint.convert(loc), np.array(desc)