import numpy as np
import cv2 as cv
from feature_detection.feature_detector import FeatureDetector

def array_from_keypoints(kp):
    return np.array([k.pt for k in kp], dtype=np.float32)

class ShiTomasiDetector(FeatureDetector):
    def __init__(self, maxCorners: int = 100, qualityLevel: float = 0.01, minDistance: float = 10):
        self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance

    def detect_features(self, image: np.ndarray) -> np.ndarray:
        # If the image is not in grayscale, convert it
        if len(image.shape) > 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp = cv.goodFeaturesToTrack(image, self.maxCorners, self.qualityLevel, self.minDistance)
        feature_array = np.array([k.pt for k in kp], dtype=np.float32)
        return feature_array