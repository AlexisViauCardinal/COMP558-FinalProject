from optical_flow.optical_flow import OpticalFlow
from scipy.spatial import KDTree
import numpy as np
from collections import deque
from optical_flow.bounding_box import BoundingBox


class Gu(OpticalFlow):

    def __init__(self):

        # general configuration
        self.number_frames = 10
        self.kd_trees = deque(maxlen = self.number_frames)

        # __compute_f params
        self._lambda = 2/3

        # __compute_k params
        self.gamma = 1
        self.position_drift = 1
        self.height = 1
        self.width = 1
        self.aspect_ratio = 1
        pass

    def track_frame(self, previous_frame, next_frame, previous_points, previous_points_desc = None, next_points = None, next_points_desc = None):
        pass

    def __compute_argmax_w(self, ok_1, wk_1, i_k) -> BoundingBox:
        '''
            Compute the best window using Efficient Subwindow Search

            ok_1    : KDTree (scipy.spatial.KDTree) storing the points in the previous frame
            wk_1    : BoundingBox representing the selected frame at the previous step
            i_k     : Current frame image intensity

            Returns a bounding box maximizing the utility function
        '''



        pass

    def __compute_f(self, a : np.ndarray, b : KDTree, c : KDTree, _lambda : float = None) -> list[bool]:
        '''
            Computes the F set according to Gu 2011

            a : Set of keypoint descriptors in an (n, m) array (numpy.ndarray) where are n points of dimension m
            b : Matching set of points (scipy.spatial.KDTree)
            c : Discriminating set of points (scipy.spatial.KDTree)

            Returns a logical array of points selected in a
        '''

        if _lambda is None:
            _lambda = self._lambda
        
        _, match_b = b.query(a)
        _, match_c = c.query(a)

        diff_b = a - b.data[match_b]
        diff_c = a - c.data[match_c]
        
        return np.linalg.norm(diff_b) < _lambda * np.linalg.norm(diff_c)
    
    def __compute_theta(self, window : BoundingBox, keypoints_location : np.ndarray) -> list[bool]:
        '''
            Computes the points contained within a window
            TODO move this to global function

            window : window (BoundingBox) use to discriminate point location
            keypoints_location: set of keypoints location as an (n, 2) array (numpy.ndarray)

            Returns a logical array of points contained in keypoints_location
        '''

        x_fit = np.logical_and(window.x <= keypoints_location[:, 0], keypoints_location[:, 0] <= window.x + window.w)
        y_fit = np.logical_and(window.y <= keypoints_location[:, 1], keypoints_location[:, 1] <= window.y + window.h)

        return np.logical_and(x_fit, y_fit)

    def __compute_k(self, w_a : BoundingBox, w_b : BoundingBox) -> float:
        '''
            Compute the motion penalty associated from moving the bounding box from w_a to w_b

            w_a : original bounding box
            w_b : subsequent bounding_box

            Returns the score (float), greater than 0, lower is better.
        '''

        # TODO compute centroid for real, taking into account width and height
        centroid = np.linalg.norm((w_a.x - w_b.x, w_a.y - w_b.y))
        height = np.abs(w_a.h - w_b.h)
        width = np.abs(w_a.w - w_b.w)
        s = np.max((w_a.h/w_a.w - w_b.h/w_b.w, w_a.w/w_a.h - w_b.w/w_b.h))

        return self.gamma * (self.position_drift * centroid + self.height * height + self.width * width + self.aspect_ratio * s)