import numpy as np
import abc

class FeatureDetector:

    @abc.abstractmethod
    def detect_features(self, image: np.ndarray) -> np.ndarray:
        """
        image : N*M RGB image (numpy.ndarray)
        
        Returns a list of keypoints (numpy.ndarray)
        """
        raise NotImplemented