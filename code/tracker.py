import numpy as np
import cv2 as cv
from optical_flow.bounding_box import BoundingBox
from optical_flow.bounding_box import drotrack_bbox_init
from optical_flow.bounding_box import drotrack_bbox_step
from optical_flow.bounding_box import center_to_bbox
from optical_flow.bounding_box import scale_bounding_box
from optical_flow.bounding_box import expand_bounding_box
from optical_flow.bounding_box import bound_bounding_box
from optical_flow.bounding_box import subset_points
from feature_detection.feature_detector import FeatureDetector
from feature_description.feature_descriptor import FeatureDescriptor
from optical_flow.optical_flow import OpticalFlow
from optical_flow.gu import Gu
from segmentation.image_segmenter import ImageSegmenter
from segmentation.segmentation_utils import cleanup
from segmentation.segmentation_utils import image_bbox
from typing import Dict, Tuple

class Tracker():

    def __init__(self, 
                 first_frame : np.ndarray,
                 initial_bbox : BoundingBox,
                 feature_detector : FeatureDetector,
                 optical_flow : OpticalFlow,
                 feature_descriptor : FeatureDescriptor,
                 segmenter : ImageSegmenter,
                 gu_params : Dict):

        # Saving parameters
        self.previous_frame = first_frame
        self.previous_bbox = initial_bbox

        self.feature_detector = feature_detector
        self.optical_flow = optical_flow

        self.feature_descriptor = feature_descriptor
        self.segmenter = segmenter

        # Statistics
        self.frame_number = -1
        self.recovery_moment = []
        self.lk_errors = []

        # Optical flow tracker
        ## Initialize optical flow points
        self.points = self.feature_detector.detect_features(self.previous_frame)
        self.points = subset_points(self.points, self.previous_bbox)

        ## bounding box properties
        self.bbox_stats = drotrack_bbox_init(self.previous_frame, self.points, self.previous_bbox)
        print(self.bbox_stats)

        # Online Classifier Tracker
        ## Trigger parameter
        self.gu_frequency = 2
        self.last_recovery = 0
        self.point_expansion_search = 1.5
        self.min_points_ratio = 0.9
        self.error_trigger = 3

        self.min_points = self.min_points_ratio * self.points.shape[0]


        ## Search parameters
        self.recovery_expansion = 5
        self.min_area_ratio = 1/100

        self.gu = Gu(self.previous_frame, self.previous_bbox, self.feature_descriptor, **gu_params)


    def track(self, frame : np.ndarray) -> BoundingBox:
        
        self.frame_number = self.frame_number + 1

        points_count, error, old_points, new_points = self.optical_flow.track_frame(self.previous_frame, frame, self.points)

        self.lk_errors.append(error)

        greater_bbox = expand_bounding_box(self.previous_bbox, self.point_expansion_search)

        time_for_udpate = (self.frame_number - self.last_recovery) % self.gu_frequency == 0
        need_recovery = points_count < self.min_points
        need_recovery = need_recovery or subset_points(new_points, greater_bbox).shape[0] < self.min_points
        need_recovery = need_recovery or np.mean(error) > self.error_trigger

        time_for_udpate = False
        need_recovery = False

        if time_for_udpate or need_recovery:
            self.last_recovery = self.frame_number
            tentative_bbox, c_score, points =  self.__recover_bbox(frame, 
                                                                   full_recovery = need_recovery, 
                                                                   previous_bbox = None)

            if need_recovery:
                new_bbox = tentative_bbox
                new_points = points

                self.bbox_stats = drotrack_bbox_init(self.previous_frame, points, self.previous_bbox)
                # self.min_points = self.min_points_ratio * new_points.shape[0]

            else:
                bbox_center, self.bbox_stats = drotrack_bbox_step(frame, self.previous_bbox, new_points, self.bbox_stats)
                new_bbox = center_to_bbox(bbox_center[0], bbox_center[1], self.previous_bbox.w, self.previous_bbox.h)

        else :
            bbox_center, self.bbox_stats = drotrack_bbox_step(frame, self.previous_bbox, new_points, self.bbox_stats)
            new_bbox = center_to_bbox(bbox_center[0], bbox_center[1], self.previous_bbox.w, self.previous_bbox.h)

        # Update internal values
        self.previous_frame = frame.copy()
        self.points = new_points.copy()
        self.previous_bbox = new_bbox

        return new_bbox
    
    def __recover_bbox(self, frame : np.ndarray, full_recovery : bool = True, previous_bbox = None) -> Tuple[BoundingBox, float, np.ndarray]:

        points = None

        # Use the Online Classifier Tracker as a baseline
        best_bbox, best_score = self.gu.track_frame(frame,
                                                    stateless = not full_recovery,
                                                    previous_bbox = previous_bbox)

        # Alleviate penalty with segmentation (time consuming)
        if full_recovery:
            a_max = tuple(np.array(frame.shape[0:2]) - 1)

            search_range = expand_bounding_box(self.previous_bbox, self.recovery_expansion)
            search_range = bound_bounding_box(search_range, (1, 1), a_max)

            search_area = frame[search_range.y:search_range.y + search_range.h, search_range.x:search_range.x + search_range.w]
            
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

            # Update tracker with best guess
            self.gu.track_frame(frame, previous_bbox = best_bbox, stateless = False)

            expanded_bbox = expand_bounding_box(best_bbox, 1.3)

            # Update optical flow features
            detected_points = self.feature_detector.detect_features(frame)
            points = subset_points(detected_points, expanded_bbox)

            if len(points) == 0:
                dist_cx = np.abs(detected_points[:, 0] - best_bbox.cx)
                dist_cy = np.abs(detected_points[:, 1] - best_bbox.cy)

                dist_x = np.clip(dist_cx - best_bbox.w / 2, 0, np.inf)
                dist_y = np.clip(dist_cy - best_bbox.h / 2, 0, np.inf)

                



        return best_bbox, best_score, points