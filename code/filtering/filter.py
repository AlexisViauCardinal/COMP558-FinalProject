import abc
import numpy as np

class Filter():

    @abc.abstractmethod
    def predict(self, reading, input = None) -> np.ndarray:
        raise NotImplemented
    