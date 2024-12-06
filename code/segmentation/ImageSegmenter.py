import abc
import numpy as np

class ImageSegmenter:

    @abc.abstractmethod
    def segment_image(self, image: np.ndarray, previous_guess: np.ndarray = None) -> np.ndarray:
        """
        image : N*M RGB image (numpy.ndarray)
        previous_guess (Optional): N*M array with guessed classification (numpy.ndarray, dtype=int)
        
        Returns an N*M classification by int values (numpy.ndarray, dtype=int)
        """
        raise NotImplemented