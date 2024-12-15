import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt
from feature_detection.feature_detector import FeatureDetector

def array_from_keypoints(kp):
    return np.array([k.pt for k in kp], dtype=np.float32)

class ORBDetector(FeatureDetector):
    def __init__(self):
        self.orb = cv.ORB_create()

    def detect_features(self, image: np.ndarray) -> np.ndarray:
        # If the image is not in grayscale, convert it
        if len(image.shape) > 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp = self.orb.detect(image, None)
        kp, des = self.orb.compute(image, kp)
        feature_array = array_from_keypoints(kp)
        return feature_array
