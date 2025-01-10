import numpy as np
import cv2 as cv
from feature_detection.feature_detector import FeatureDetector
from typing import Dict

class ShiTomasiDetector(FeatureDetector):
    def __init__(self, params : Dict = {"maxCorners": 100, "qualityLevel": 0.01, "minDistance": 10}):
        self.params = params

    def detect_features(self, image: np.ndarray) -> np.ndarray:
        
        if len(image.shape) > 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        kp = cv.goodFeaturesToTrack(image, **self.params)

        return np.squeeze(kp)