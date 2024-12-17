import numpy as np
import cv2 as cv
from feature_description.feature_descriptor import FeatureDescriptor

class ORBDescriptor(FeatureDescriptor):

    def __init__(self, params={}):
        """
        Initializes the ORB descriptor with optional parameters.
        """
        super().__init__()
        self.orb_obj = cv.ORB_create(**params)

    def detect_features(self, image: np.ndarray, mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Computes distinctive ORB features.

        image: a BGR image as an N*M*3 array (np.ndarray)
        mask: an N*M array masking possible point location (1 is to include, 0 is to exclude)

        Returns:
            - points: (N, 2) array (np.ndarray) containing the keypoint locations.
            - descriptors: (N, M) array (np.ndarray) containing the associated binary descriptors.
            - sizes: (N,) array (np.ndarray) containing the size of each keypoint.
        '''
        if len(image.shape) == 3:  # Image has 3 channels (BGR)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:  # Image is already grayscale
            gray = image

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb_obj.detectAndCompute(gray, mask=mask)

        # Convert keypoints to numpy arrays
        points = cv.KeyPoint.convert(keypoints)
        sizes = np.array([kp.size for kp in keypoints])  # Keypoint sizes

        return points, np.array(descriptors), sizes