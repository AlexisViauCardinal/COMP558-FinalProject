import numpy as np
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
from typing import Any, Tuple, Dict

class Tracker():

    def __init__(self, 
                 first_frame : np.ndarray,
                 initial_bbox : BoundingBox,
                 feature_detector : FeatureDetector,
                 optical_flow : OpticalFlow,
                 feature_descriptor : FeatureDescriptor,
                 segmenter : ImageSegmenter,
                 gu_params : Dict):

        # saving parameters
        self.previous_frame = first_frame
        self.previous_bbox = initial_bbox
        self.feature_detector = feature_detector
        self.optical_flow = optical_flow

        # statistics
        self.frame_number = -1
        self.recovery_moment = []
        self.segmentation_failed = []

        # Initialize optical flow points
        self.points = self.feature_detector.detect_features(self.previous_frame)
        self.points = subset_points(self.points, self.previous_bbox)

        # bounding box properties
        self.bbox_stats = drotrack_bbox_init(self.previous_frame, self.points, self.previous_bbox)


    def track(self, frame : np.ndarray) -> BoundingBox:
        
        self.frame_number = self.frame_number + 1

        points_count, error, old_points, new_points = self.optical_flow.track_frame(self.previous_frame, frame, self.points)

        bbox_center, self.bbox_stats = drotrack_bbox_step(frame, self.previous_bbox, new_points, self.bbox_stats)
        new_bbox = center_to_bbox(bbox_center[0], bbox_center[1], self.previous_bbox.w, self.previous_bbox.h)
        

        # Update internal values
        self.previous_frame = frame.copy()
        self.points = new_points.copy()
        self.previous_bbox = new_bbox

        return new_bbox
