import numpy as np
from optical_flow.bounding_box import BoundingBox
from optical_flow.bounding_box import expand_bounding_box
from optical_flow.bounding_box import bound_bounding_box
from optical_flow.points_utils import compute_bbox
from optical_flow.points_utils import subset_points
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
        self.frame_dimension = first_frame.shape
        self.full_frame_bbox = BoundingBox(0, 0, self.frame_dimension[1], self.frame_dimension[0])

        # Online Classifier Tracker
        ## Trigger parameter
        self.gu_frequency = 3
        self.last_recovery = 0
        self.point_expansion_search = 1
        self.min_points_ratio = 0.5
        self.error_trigger = 4

        self.min_points = self.min_points_ratio * self.points.shape[0]
        self.abs_min_points = 3

        self.recovery_missing_point_expansion = 1.1


        ## Search parameters
        self.recovery_expansion = 5
        self.min_area_ratio = 1/100

        self.gu = Gu(self.previous_frame, self.previous_bbox, self.feature_descriptor, **gu_params)


    def track(self, frame : np.ndarray) -> BoundingBox:
        
        self.frame_number = self.frame_number + 1

        points_count, error, old_points, new_points = self.optical_flow.track_frame(self.previous_frame, frame.copy(), self.points)

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
                                                                   previous_bbox = self.previous_bbox if not need_recovery else None)

        if need_recovery:
            new_bbox = tentative_bbox
            new_points = points

        else:
            new_bbox, bbox_error = compute_bbox(points, self.full_frame_bbox)


        # Update internal values
        self.previous_frame = frame.copy()
        self.points = new_points.copy()
        self.previous_bbox = new_bbox

        return new_bbox
    
    def __recover_bbox(self, frame : np.ndarray, full_recovery : bool = True, previous_bbox = None) -> Tuple[BoundingBox, float, np.ndarray]:

        points = None

        # Use the Online Classifier Tracker as a baseline
        best_bbox, best_score = self.gu.track_frame(frame,
                                                    stateless = full_recovery,
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

                dist = np.linalg.norm(np.vstack([dist_x, dist_y]).T, axis = 1)

                min = np.argmin(dist)

                delta_x = (detected_points[min, 0] - best_bbox.x) * self.recovery_missing_point_expansion
                delta_y = (detected_points[min, 1] - best_bbox.y) * self.recovery_missing_point_expansion

                if delta_x < 0:
                    target_x = int(np.floor(best_bbox.x + delta_x))
                else :
                    target_x = int(np.floor(best_bbox.x))

                if delta_y < 0:
                    target_y = int(np.floor(best_bbox.y + delta_y))
                else :
                    target_y = int(np.floor(best_bbox.y))

                target_w = int(np.ceil(bbox.w + np.abs(delta_x)))
                target_h = int(np.ceil(bbox.h + np.abs(delta_y)))

                best_bbox = BoundingBox(target_x, target_y, target_w, target_h)
                points = subset_points(detected_points, best_bbox)

                best_score = np.inf

        return best_bbox, best_score, points