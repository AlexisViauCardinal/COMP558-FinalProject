from optical_flow.optical_flow import OpticalFlow
import cv2 as cv

class LucasKanade(OpticalFlow):

    def __init__(self, lk_params = {}):
        self.lk_params = lk_params


    def track_frame(self, previous_frame, next_frame, previous_points, previous_points_desc = None, next_points = None, next_points_desc = None):
        previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        return cv.calcOpticalFlowPyrLK(previous_gray, next_gray, previous_points, next_points, **self.lk_params)

