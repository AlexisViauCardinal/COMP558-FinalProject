import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from collections import deque
from optical_flow.bounding_box import BoundingBox
from optical_flow.ess import FuzzyBoundingBox
from optical_flow.ess import get_largest_bounding_box
from optical_flow.ess import get_smallest_bounding_box
from optical_flow.ess import ess_search
from feature_description.feature_descriptor import FeatureDescriptor


class Gu:

    def __init__(self, first_frame : np.ndarray, 
                 bounding_box : BoundingBox, 
                 descriptor : FeatureDescriptor, 
                 frame_buffer : int = 10,
                 _lambda : float = 2/3,
                 gamma : float = 0.1,
                 gamma_drift : float = 1.0,
                 gamma_width : float = 1.0,
                 gamma_height : float = 1.0,
                 gamma_aspect_ratio : float = 1.0,
                 gamma_area : float = 0.0 ):

        # general configuration
        self.number_frames = frame_buffer
        self.kd_trees = deque(maxlen = self.number_frames)
        # self.background_tree = deque(maxlen = self.number_frames)

        # __compute_f params
        self._lambda = _lambda

        # __compute_k params
        self.gamma = gamma
        self.gamma_drift = gamma_drift
        self.gamma_width = gamma_width
        self.gamma_height = gamma_height
        self.gamma_aspect_ratio = gamma_aspect_ratio
        self.gamma_area = gamma_area

        # feature descriptor
        self.descriptor = descriptor
        
        # first frame init
        points_loc, points_desc, points_size = self.descriptor.detect_features(first_frame)

        theta = self.__compute_theta(bounding_box, points_loc)

        self.kd_trees.append(KDTree(points_desc[theta, :]))
        # self.background_tree.append(KDTree(points_desc[~theta, :]))
        self.background_tree = KDTree(points_desc[~theta, :])

        self.previous_bbox = bounding_box

    def track_frame(self, next_frame : np.ndarray, previous_bbox : BoundingBox = None, stateless : bool = False) -> tuple[BoundingBox, float]:
        '''
            Computes one iteration of the Gu object tracker.

            next_frame  : next frame on which to perform tracking
            previous_bbox : bounding box (BoundingBox) overriding the state variable
            stateless : prevent from updating the state of the Gu object
            
            Returns the new bounding box
        '''
        if previous_bbox is None:
            previous_bbox = self.previous_bbox

        points_loc, points_desc, points_size = self.descriptor.detect_features(next_frame)
        
        foreground = np.full((points_loc.shape[0], ), False)

        # for i in range(np.min((self.number_frames, len(self.kd_trees), len(self.background_tree)))):
        for tree in self.kd_trees:
            # iter_res = self.__compute_f(points_desc, self.kd_trees[i], self.background_tree[i])
            iter_res = self.__compute_f(points_desc, tree, self.background_tree)
            foreground = np.logical_or(foreground, iter_res)

        w, score = self.__compute_argmax_w(points_loc, points_size, foreground, previous_bbox, next_frame)
        theta = self.__compute_theta(w, points_loc)
        f_set = points_desc[np.logical_and(foreground, theta), :]
        f_not_set = points_desc[~np.logical_and(foreground, theta), :]
        
        if not stateless:
            # update foreground
            self.kd_trees.append(KDTree(f_set))

            # update background
            # self.background_tree.append(KDTree(f_not_set))
            self.background_tree = KDTree(f_not_set)

            self.previous_bbox = w


        asdf = points_loc[foreground, :]
        for j in range(asdf.shape[0]):
            next_frame = cv.circle(next_frame, np.int_(asdf[j]), 3, (0, 255, 0), -1)

        asdf = points_loc[~foreground, :]
        for j in range(asdf.shape[0]):
            next_frame = cv.circle(next_frame, np.int_(asdf[j]), 3, (0, 0, 255), -1)

        return w, score, next_frame

    def __compute_argmax_w(self,
                           keypoints_loc : np.ndarray, 
                           keypoints_size : np.ndarray, 
                           keypoints_in_foreground : np.ndarray, 
                           wk_1 : BoundingBox, 
                           i_k : np.ndarray) -> tuple[BoundingBox, float]:
        '''
            Compute the best window using Efficient Subwindow Search

            keypoints_loc           : Location of points (x, y) as an array (np.ndarray)
            keypoints_in_foreground : Logical array having true whenever a point is thought to be in the tracked object
            wk_1    : BoundingBox representing the selected frame at the previous step
            i_k     : Current frame image intensity

            Returns a bounding box maximizing the utility function
        '''

        shape = i_k.shape[0:2]

        def f_hat(fuzzy : FuzzyBoundingBox) -> float:

            # Compute kappa
            ## naive assumption that best fitting window within boundaries actually minimizes the error
            x = np.clip(wk_1.x, fuzzy.l.low, fuzzy.l.high)
            y = np.clip(wk_1.y, fuzzy.b.low, fuzzy.b.high)
            w = np.clip(np.clip(wk_1.x + wk_1.w, fuzzy.r.low, fuzzy.r.high), 0, shape[1]) - x
            h = np.clip(np.clip(wk_1.y + wk_1.h, fuzzy.t.low, fuzzy.t.high), 0, shape[0]) - y

            if w == 0 or h == 0:
                return -np.inf

            wk = BoundingBox(x, y, w, h)

            kappa = self.__compute_kappa(wk_1, wk)

            # Compute largest and smallest possible box
            large = get_largest_bounding_box(fuzzy)
            small = get_smallest_bounding_box(fuzzy)

            # Compute the positive points
            # subset_large_x = np.logical_and(keypoints_loc[:, 0] - keypoints_size >= large.x, keypoints_loc[:, 0] + keypoints_size < large.x + large.w)
            # subset_large_y = np.logical_and(keypoints_loc[:, 1] - keypoints_size >= large.y, keypoints_loc[:, 1] + keypoints_size < large.y + large.h)
            theta_plus = self.__compute_theta(large, keypoints_loc[keypoints_in_foreground, :])
            points_plus = np.sum(theta_plus)

            # Compute the negative points
            theta_minus = self.__compute_theta(small, keypoints_loc[~keypoints_in_foreground, :])
            points_minus = np.sum(theta_minus)

            return points_plus - points_minus - kappa

        
        search_bbox = BoundingBox(0, 0, shape[1], shape[0])

        return ess_search(search_bbox, f_hat)

    def __compute_f(self, a : np.ndarray, b : KDTree, c : KDTree, _lambda : float = None) -> np.ndarray:
        '''
            Computes the F set according to Gu 2011

            a : Set of keypoint descriptors in an (n, m) array (numpy.ndarray) where are n points of dimension m
            b : Matching set of points (scipy.spatial.KDTree)
            c : Discriminating set of points (scipy.spatial.KDTree)

            Returns a logical array of points selected in a
        '''

        if _lambda is None:
            _lambda = self._lambda

        if b.n == 0 or c.n == 0:
            return np.full((a.shape[0]), False)
        
        diff_b, _ = b.query(a)
        diff_c, _ = c.query(a)
        
        return diff_b < _lambda * diff_c
    
    def __compute_theta(self, window : BoundingBox, keypoints_location : np.ndarray) -> np.ndarray:
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

    def __compute_kappa(self, w_a : BoundingBox, w_b : BoundingBox) -> float:
        '''
            Compute the motion penalty associated from moving the bounding box from w_a to w_b

            w_a : original bounding box
            w_b : subsequent bounding_box

            Returns the score (float), greater than 0, lower is better.
        '''

        centroid = self.gamma_drift * np.linalg.norm((w_a.cx - w_b.cx, w_a.cy - w_b.cy))
        width = self.gamma_width * np.abs(w_a.w - w_b.w)
        height = self.gamma_height * np.abs(w_a.h - w_b.h)
        s = self.gamma_aspect_ratio * np.max((np.abs(w_a.h/w_a.w - w_b.h/w_b.w), np.abs(w_a.w/w_a.h - w_b.w/w_b.h)))

        # added parameter
        area_change = self.gamma_area * np.sqrt(np.abs(w_a.w * w_a.h - w_b.w * w_b.h))

        return self.gamma * (centroid + height + width + s + area_change)