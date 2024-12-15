import cv2 as cv
from optical_flow.optical_flow import OpticalFlow
import numpy as np

class LucasKanade(OpticalFlow):

    def __init__(self, lk_params = {}):
        self.lk_params = lk_params


    def track_frame(self, previous_frame : np.ndarray, next_frame : np.ndarray, previous_points : np.ndarray, next_points : np.ndarray = None):
        '''

        return (number of points, error, old points, new points)
        '''
        
        previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(previous_gray, next_gray, previous_points, next_points, **self.lk_params)

        if p1 is None:
            return 0, np.inf, None, None
        
        good_new = p1[st==1]
        good_old = previous_points[st==1]

        return np.sum(st), err, good_old, good_new

