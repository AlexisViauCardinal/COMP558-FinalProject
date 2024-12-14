from filtering.filter import Filter
import numpy as np

class KalmanFilter(Filter):

    def __init__(self, x, F, H, P = None, Q = None, R = None, B = None):
        super().__init__()

        assert len(F.shape) == 2 and F.shape[0] == F.shape[1]
        self.n = np.min(F.shape)

        if P is None:
            P = np.identity(self.n)

        if Q is None:
            Q = np.identity(self.n)

        if R is None:
            R = np.identity(H.shape[0])

        if B is None:
            B = np.identity(self.n)

        self.x = x

        self.F = F  # State transition model
        self.H = H  # Observation model
        self.P = P  # Posteriori estimate covariance matrix
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Observation covariance
        self.B = B  # Control input

    def __kalman_predict(self, input):
        self.x = self.F @ self.x + self.B @ input
        self.P = self.F @ self.P @ np.transpose(self.F) + self.Q

        return self.x

    def __kalman_update(self, reading):
        
        # innovation
        y = reading - self.H @ self.x

        # innovation covariance
        S = self.H @ self.P @ np.transpose(self.H) + self.R

        # Kalman gain
        self.K = self.P @ np.transpose(self.H) @ np.linalg.inv(S)

        # Updated estimate
        self.x = self.x + self.K @ y

        # Updated estimate covariance
        self.P = (np.identity(self.n) - self.K @ self.H) @ self.P

        # Residual
        y_kk = reading - self.H @ self.x    # TODO useless?

        return self.x

    def predict(self, reading, input = None):

        if input is None:
            input = np.zeros((self.n, 1))

        self.__kalman_predict(input)
        return self.__kalman_update(reading)