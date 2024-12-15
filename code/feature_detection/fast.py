import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt
from feature_detection.feature_detector import FeatureDetector

def array_from_keypoints(kp):
    return np.array([k.pt for k in kp], dtype=np.float32)

class FASTDetector(FeatureDetector):
    def __init__(self, threshold: int = 100):
        self.fast = cv.FastFeatureDetector_create()
        self.fast.setThreshold(threshold)

    def detect_features(self, image: np.ndarray) -> np.ndarray:
        # If the image is not in grayscale, convert it
        if len(image.shape) > 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp = self.fast.detect(image, None)
        feature_array = array_from_keypoints(kp)
        return feature_array