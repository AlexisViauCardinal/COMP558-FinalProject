import abc
import numpy as np

class OpticalFlow:

    @abc.abstractmethod
    def track_frame(self, previous_frame : np.ndarray, next_frame : np.ndarray, previous_points : np.ndarray, next_points : np.ndarray = None):
        raise NotImplemented