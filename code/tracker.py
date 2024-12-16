import numpy as np
from optical_flow.bounding_box import BoundingBox
from optical_flow.bounding_box import drotrack_bbox_init
from optical_flow.bounding_box import drotrack_bbox_step
from optical_flow.bounding_box import center_to_bbox
from feature_detection.feature_detector import FeatureDetector
from feature_description.feature_descriptor import FeatureDescriptor
from optical_flow.optical_flow import OpticalFlow
from optical_flow.gu import Gu
from segmentation.image_segmenter import ImageSegmenter
from segmentation.segmentation_utils import cleanup
from segmentation.segmentation_utils import image_bbox
import cv2 as cv


def subset_points(points : np.ndarray, bbox : BoundingBox):
    subset_x = np.logical_and(points[:, 0, 0] >= bbox.x, points[:, 0, 0] <= bbox.x + bbox.w)
    subset_y = np.logical_and(points[:, 0, 1] >= bbox.y, points[:, 0, 1] <= bbox.y + bbox.h)
    subset = np.logical_and(subset_x, subset_y)

    return points[subset, ...]

class Tracker():

    def __init__(self, first_frame : np.ndarray, initial_bbox : BoundingBox, feature_detector : FeatureDetector, optical_flow : OpticalFlow, feature_descriptor : FeatureDescriptor, segmenter : ImageSegmenter, gu_params = {}):

        # saving parameters
        self.previous_frame = first_frame
        self.previous_bbox = initial_bbox
        self.feature_detector = feature_detector
        self.previous_frame = first_frame
        self.optical_flow = optical_flow

        # initializing points
        self.points = feature_detector.detect_features(first_frame)
        self.points = subset_points(self.points, initial_bbox)
        self.gu = Gu(first_frame, initial_bbox, feature_descriptor)

        self.min_points = 0.1 * self.points.shape[0]

        # recovery parameters
        self.gu_frequence = 1
        self.gu_iter = 0
        self.recovery_expansion = 5

        self.segmenter = segmenter
        # self.min_area_ratio = 1/150 # should be at least 1/100
        self.min_area_ratio = 1

        self.frame_number = -1
        self.recovery_moment = []
        self.segmentation_failed = []

        # bounding box properties
        self.bbox_stats = drotrack_bbox_init(self.previous_frame, self.points, self.previous_bbox)



    def track(self, frame : np.ndarray) -> BoundingBox:
        
        self.frame_number = self.frame_number + 1

        pt_count, err, points_old, points_new = self.optical_flow.track_frame(self.previous_frame, frame, self.points)
        

        # TODO check for err
        if pt_count < self.min_points:
            self.recovery_moment.append(self.frame_number)
            new_bbox, _ = self.__recover_bbox(frame)

            points_new = self.feature_detector.detect_features(frame)
            points_new = subset_points(points_new, new_bbox)

        else :
            points_new = points_new.reshape(-1, 1, 2)
            bbox_center, self.bbox_stats = drotrack_bbox_step(frame, self.previous_bbox, points_new, self.bbox_stats)
            new_bbox = center_to_bbox(bbox_center[0], bbox_center[1], self.previous_bbox.w, self.previous_bbox.h)
            # TODO SCALE WITH CAMSHIFT

        # do last to avoid over writting states
        self.gu_iter = self.gu_iter + 1

        if self.gu_iter % self.gu_frequence == 0:
            self.gu_iter = 0
            self.gu.track_frame(frame)

        self.previous_bbox = new_bbox
        self.points = points_new

        return new_bbox

    def __recover_bbox(self, frame : np.ndarray) -> BoundingBox:

        a_max = tuple(np.array(frame.shape[0:2]) - 1)

        min_x = self.previous_bbox.cx - self.previous_bbox.w * self.recovery_expansion 
        max_x = self.previous_bbox.cx + self.previous_bbox.w * self.recovery_expansion 
        
        min_y = self.previous_bbox.cy - self.previous_bbox.h * self.recovery_expansion 
        max_y = self.previous_bbox.cy + self.previous_bbox.h * self.recovery_expansion 

        search_range = np.clip(((min_x, min_y), (max_x, max_y)), a_min = (0, 0), a_max = a_max)
        search_area = frame[search_range[0, 1]:search_range[1, 1], search_range[0, 0]:search_range[1, 0]]
        
        w_h = search_range[1, :] - search_range[0, :]
        area = w_h[0] * w_h[1]

        segmented = self.segmenter.segment_image(search_area)
        post_processed = cleanup(segmented)

        regions = image_bbox(post_processed, min_area=int(self.min_area_ratio * area))

        best_bbox, best_score = None, np.inf
        for region in regions:
            minr, minc, maxr, maxc = region.bbox

            test_bbox = BoundingBox(min_x + minc, min_y + minr, maxc - minc, maxr - minr)

            bbox, error = self.gu.track_frame(frame, previous_bbox = test_bbox, stateless = True)

            if best_score > error:
                best_score = error
                best_bbox = bbox

        self.segmentation_failed.append(self.frame_number)
        bbox, error = self.gu.track_frame(frame, stateless = True)
        
        if best_score > error:
            best_score = error
            best_bbox = bbox

        self.gu.track_frame(frame, previous_bbox = best_bbox, stateless = False)

        return bbox, error

'''
what we do

* track with camshift
* track with lucas-kanada
* if they are different, segment, call gu


or track with LK, change scale with meanshift.
always use LK coordinate for both next step lK and meanshift


or dense optical flow -> camshift

'''