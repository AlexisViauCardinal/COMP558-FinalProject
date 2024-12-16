import abc
import numpy as np

class FeatureDescriptor:

    @abc.abstractmethod
    def detect_features(self, image : np.ndarray, mask : np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
            Computes distinctive features to track through optical flow

            image: an BGR image as an N*M*3 array (np.ndarray)
            mask: an N*M array masking possible point location (1 is to include, 0 is to exclude)

            Returns an (N, 2) array (np.ndarray) containing the points location and associated feature descriptors as an (N, M) array
        '''
        pass