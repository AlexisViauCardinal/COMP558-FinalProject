from filtering.filter import Filter
import numpy as np

class LowPassFilter(Filter):

    def __init__(self, init, alpha):
        
        assert np.all(0 <= alpha) and np.all(alpha <= 1)

        self.val = init
        self.default_input = np.zeros_like(init)
        self.alpha = alpha

    def predict(self, reading, input = None):

        if input is None:
            input = self.default_input

        self.val = self.alpha * self.val + (1 - self.alpha) * (reading - input)

        return self.val