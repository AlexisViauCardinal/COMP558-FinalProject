import abc 
import numpy as np

class ScaleCorrection(abc.ABC):
    @abc.abstractmethod
    def correct_scale(self, frame: np.ndarray):
        """
        frame : N*M RGB image (numpy.ndarray)
        
        Returns an N*M RGB image (numpy.ndarray)
        """
        raise NotImplementedError
    