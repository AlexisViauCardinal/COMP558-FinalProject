import numpy as np
from scipy.spatial import KDTree
from collections import deque
from optical_flow.bounding_box import BoundingBox
from optical_flow.ess import FuzzyBoundingBox
from optical_flow.ess import get_largest_bounding_box
from optical_flow.ess import get_smallest_bounding_box
from optical_flow.ess import ess_search
from feature_description.feature_descriptor import FeatureDescriptor


class Gu:

    def __init__(self, first_frame : np.ndarray, bounding_box : BoundingBox, descriptor : FeatureDescriptor):

        # general configuration
        self.number_frames = 60
        self.kd_trees = deque(maxlen = self.number_frames)
        self.background_tree = deque(maxlen = self.number_frames)
        # self.background_tree = None

        # __compute_f params
        self._lambda = 4/5

        # __compute_k params
        self.gamma = 0.1
        self.position_drift = 1
        self.height = 1
        self.width = 1
        self.aspect_ratio = 1

        # feature descriptor
        self.descriptor = descriptor
        
        # first frame init
        points_loc, points_desc = self.descriptor.detect_features(first_frame)
        theta = self.__compute_theta(bounding_box, points_loc)

        self.kd_trees.append(KDTree(points_desc[theta, :]))
        self.background_tree.append(KDTree(points_desc[~theta, :]))
        # self.background_tree = KDTree(points_desc[~theta, :])

        self.previous_bbox = bounding_box

    def track_frame(self, next_frame : np.ndarray, previous_bbox : BoundingBox = None) -> tuple[BoundingBox, float]:
        '''
            Does one iteration of the Gu object tracker.

            next_frame  : next frame on which to perform tracking
            
            Returns the new bounding box
        '''
        if previous_bbox is None:
            previous_bbox = self.previous_bbox

        points_loc, points_desc = self.descriptor.detect_features(next_frame)
        
        foreground = np.full((points_loc.shape[0], ), False)

        for i in range(np.min((self.number_frames, len(self.kd_trees), len(self.background_tree)))):
        # for tree in self.kd_trees:
            iter_res = self.__compute_f(points_desc, self.kd_trees[i], self.background_tree[i])
            # iter_res = self.__compute_f(points_desc, tree, self.background_tree)
            foreground = np.logical_or(foreground, iter_res)

        w, score = self.__compute_argmax_w(points_loc, foreground, previous_bbox, next_frame)
        theta = self.__compute_theta(w, points_loc)
        f_set = points_desc[np.logical_and(foreground, theta), :]
        f_not_set = points_desc[~np.logical_and(foreground, theta), :]
        
        # update foreground
        self.kd_trees.append(KDTree(f_set))

        # update background
        self.background_tree.append(KDTree(f_not_set))
        # self.background_tree = KDTree(f_not_set)

        self.previous_bbox = w

        return w, score

    def __compute_argmax_w(self, keypoints_loc : np.ndarray, keypoints_in_foreground : np.ndarray, wk_1, i_k) -> tuple[BoundingBox, float]:
        '''
            Compute the best window using Efficient Subwindow Search

            keypoints_loc           : Location of points (x, y) as an array (np.ndarray)
            keypoints_in_foreground : Logical array having true whenever a point is thought to be in the tracked object
            wk_1    : BoundingBox representing the selected frame at the previous step
            i_k     : Current frame image intensity

            Returns a bounding box maximizing the utility function
        '''

        def f_hat(fuzzy : FuzzyBoundingBox) -> float:

            # Compute largest and smallest possible box
            large = get_largest_bounding_box(fuzzy)
            small = get_smallest_bounding_box(fuzzy)

            # Compute the positive points
            subset_large_x = np.logical_and(keypoints_loc[:, 0] >= large.x, keypoints_loc[:, 0] < large.x + large.w)
            subset_large_y = np.logical_and(keypoints_loc[:, 1] >= large.y, keypoints_loc[:, 1] < large.y + large.h)
            subset_large = np.logical_and(subset_large_x, subset_large_y)

            large_match = keypoints_in_foreground[subset_large]
            points_plus = np.sum(np.where(large_match, 1, 0))

            # Compute the negative points
            subset_small_x = np.logical_and(keypoints_loc[:, 0] >= small.x, keypoints_loc[:, 0] < small.x + small.w)
            subset_small_y = np.logical_and(keypoints_loc[:, 1] >= small.y, keypoints_loc[:, 1] < small.y + small.h)
            subset_small = np.logical_and(subset_small_x, subset_small_y)

            small_match = keypoints_in_foreground[subset_small]
            points_minus = np.sum(np.where(small_match, 0, -1))

            # Compute kappa
            ## naive assumption that best fitting window minimizes the error

            x = np.clip(wk_1.x, fuzzy.l.low, fuzzy.l.high)
            y = np.clip(wk_1.y, fuzzy.b.low, fuzzy.b.high)
            w = np.clip(np.clip(wk_1.x + wk_1.w, fuzzy.r.low, fuzzy.r.high) - wk_1.x, 0, np.inf)
            h = np.clip(np.clip(wk_1.y + wk_1.h, fuzzy.t.low, fuzzy.t.high) - wk_1.y, 0, np.inf)

            if w == 0 or h == 0:
                return -np.inf

            wk = BoundingBox(x, y, w, h)

            kappa = self._compute_kappa(wk_1, wk)

            return points_plus + points_minus - kappa
        
        # Compute whole picture bounding box

        shape = i_k.shape[0:2]
        whole_picture_bbox = BoundingBox(0, 0, shape[1], shape[0])

        return ess_search(whole_picture_bbox, f_hat)

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
        
        _, match_b = b.query(a)
        _, match_c = c.query(a)

        diff_b = a - b.data[match_b]
        diff_c = a - c.data[match_c]
        
        return np.linalg.norm(diff_b, axis = 1) < _lambda * np.linalg.norm(diff_c, axis = 1)
    
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

    def _compute_kappa(self, w_a : BoundingBox, w_b : BoundingBox) -> float:
        '''
            Compute the motion penalty associated from moving the bounding box from w_a to w_b

            w_a : original bounding box
            w_b : subsequent bounding_box

            Returns the score (float), greater than 0, lower is better.
        '''

        centroid = np.linalg.norm((w_a.x + w_a.w/2 - w_b.x - w_b.w/2, w_a.y + w_a.h/2 - w_b.y - w_b.h/2))
        height = np.abs(w_a.h - w_b.h)
        width = np.abs(w_a.w - w_b.w)
        s = np.max((w_a.h/w_a.w - w_b.h/w_b.w, w_a.w/w_a.h - w_b.w/w_b.h))

        return self.gamma * (self.position_drift * centroid + self.height * height + self.width * width + self.aspect_ratio * s)