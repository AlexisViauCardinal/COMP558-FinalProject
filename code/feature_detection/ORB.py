import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt
# from feature_detection.FeatureDetector import FeatureDetector
from FeatureDetector import FeatureDetector

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

if __name__=="__main__":

    # Load test images
    img = cv.imread("images/IMG_0055.jpeg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv.ORB_create()

    # Find the keypoints with ORB
    kp = orb.detect(img_gray, None)

    # Compute the descriptors with ORB
    kp, des = orb.compute(img_gray, kp)

    # Draw only keypoints location, not size and orientation
    # img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    # Draw keypoints with size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # features = np.zeros((len(kp), 2))
    # for i in range(len(kp)):
    #     features[i, :] = kp[i].pt
    features = np.array([k.pt for k in kp], dtype=np.float32)
    # print(features[2:5, :])
    # plt.imshow(img2), plt.show()
    for i in features:
        x, y = i.ravel()
        print(x, y)

