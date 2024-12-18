import numpy as np
from optical_flow.bounding_box import BoundingBox
from optical_flow.bounding_box import drotrack_bbox_init
from optical_flow.bounding_box import drotrack_bbox_step
from optical_flow.bounding_box import center_to_bbox
from optical_flow.bounding_box import scale_bounding_box
from optical_flow.bounding_box import expand_bounding_box
from optical_flow.bounding_box import bound_bounding_box
from feature_detection.feature_detector import FeatureDetector
from feature_description.feature_descriptor import FeatureDescriptor
from optical_flow.optical_flow import OpticalFlow
from optical_flow.gu import Gu
from segmentation.image_segmenter import ImageSegmenter
from segmentation.segmentation_utils import cleanup
from segmentation.segmentation_utils import image_bbox
import cv2 as cv
from utils.test_utils import ErrorTracker
import utils.test_utils as T

def subset_points(points : np.ndarray, bbox : BoundingBox):
    subset_x = np.logical_and(points[:, 0, 0] >= bbox.x, points[:, 0, 0] <= bbox.x + bbox.w)
    subset_y = np.logical_and(points[:, 0, 1] >= bbox.y, points[:, 0, 1] <= bbox.y + bbox.h)
    subset = np.logical_and(subset_x, subset_y)

    return points[subset, ...]

class Tracker():

    def __init__(self, first_frame : np.ndarray, initial_bbox : BoundingBox, feature_detector : FeatureDetector, optical_flow : OpticalFlow, feature_descriptor : FeatureDescriptor, segmenter : ImageSegmenter, gu_params = {}, gt_csv = None ):

        # saving parameters
        self.previous_frame = first_frame
        self.previous_bbox = initial_bbox
        self.feature_detector = feature_detector
        self.previous_frame = first_frame
        self.optical_flow = optical_flow

        # initializing points
        self.points = feature_detector.detect_features(first_frame)
        self.points = subset_points(self.points, initial_bbox)

        self.min_points_ratio = 0.5 # how many points should still be tracked before using GU
        self.min_points = self.min_points_ratio * self.points.shape[0]

        # recovery parameters
        self.gu_frequence = 5   # frame frequency of Gu update
        self.gu_scale_down = 2      # image scaledown factor
        self.recovery_expansion = 5     # 

        self.gu = Gu(first_frame, initial_bbox, feature_descriptor)

        # segmentation
        self.segmenter = segmenter
        self.min_area_ratio = 1/150 # should be at least 1/100

        # internal counter
        self.frame_number = -1
        self.recovery_moment = []
        self.segmentation_failed = []

        print(self.points)
        # bounding box properties
        self.bbox_stats = drotrack_bbox_init(self.previous_frame, self.points, self.previous_bbox)

        # initialize error tracker
        self.E_tracker = None
        if gt_csv is not None: 
            self.E_tracker = ErrorTracker(gt_csv)


    def track(self, frame : np.ndarray) -> BoundingBox:
        
        self.frame_number = self.frame_number + 1

        pt_count, err, points_old, points_new = self.optical_flow.track_frame(self.previous_frame, frame, self.points)

        need_recover = pt_count < self.min_points
        need_recover = need_recover or points_new is None
        # TODO check for err

        if not need_recover:
            points_new = points_new.reshape(-1, 1, 2)
            greater_bbox = expand_bounding_box(self.previous_bbox, 1.5)
            points_new = subset_points(points_new, greater_bbox)

        need_recover = need_recover or pt_count < self.min_points
        
        if need_recover:
            self.recovery_moment.append(self.frame_number)
            new_bbox, _ = self.__recover_bbox(frame)

            new_bbox = expand_bounding_box(new_bbox, 1.5)

            points_new = self.feature_detector.detect_features(frame)
            points_new = subset_points(points_new, new_bbox)

            # self.min_points = self.min_points_ratio * points_new.shape[0]

        else :
            bbox_center, self.bbox_stats = drotrack_bbox_step(frame, self.previous_bbox, points_new, self.bbox_stats)
            new_bbox = center_to_bbox(bbox_center[0], bbox_center[1], self.previous_bbox.w, self.previous_bbox.h)
            # TODO SCALE WITH CAMSHIFT

        # do last to avoid over writting states
        if self.frame_number % self.gu_frequence == 0:
            self.gu.track_frame(frame)

        self.previous_bbox = new_bbox
        self.points = points_new
        self.previous_frame = frame

        # maybe update error tracker
        if self.E_tracker is not None:
            self.E_tracker.update(new_bbox)

        return new_bbox

    def __recover_bbox(self, frame : np.ndarray) -> BoundingBox:

        # frame_scale_down_size = tuple((np.array(frame.shape[0:2])/self.gu_scale_down).astype(np.int32))
        # frame_scaled_down = cv.resize(frame, frame_scale_down_size)


        best_bbox, best_score = self.gu.track_frame(frame, stateless = True)

        a_max = tuple(np.array(frame.shape[0:2]) - 1)

        search_range = expand_bounding_box(self.previous_bbox, self.recovery_expansion)
        search_range = bound_bounding_box(search_range, (1, 1), a_max)

        search_area = frame[search_range.y:search_range.y + search_range.h, search_range.x:search_range.x + search_range.w]
        if np.min(search_area.shape) == 0:
            return best_bbox, best_score
        
        area = search_range.w * search_range.h

        segmented = self.segmenter.segment_image(search_area)
        post_processed = cleanup(segmented)

        regions = image_bbox(post_processed, min_area=int(self.min_area_ratio * area))

        for region in regions:
            minr, minc, maxr, maxc = region.bbox

            test_bbox = BoundingBox(search_range.x + minc, search_range.y + minr, maxc - minc, maxr - minr)

            bbox, error = self.gu.track_frame(frame, previous_bbox = test_bbox, stateless = True)

            if best_score > error:
                best_score = error
                best_bbox = bbox

        self.gu.track_frame(frame, previous_bbox = best_bbox, stateless = False)

        return best_bbox, best_score
    
    def get_error_tracker(self):
        return self.E_tracker

'''
what we do

* track with camshift
* track with lucas-kanada
* if they are different, segment, call gu


or track with LK, change scale with meanshift.
always use LK coordinate for both next step lK and meanshift


or dense optical flow -> camshift

'''