import numpy as np
from typing import Tuple
from dataclasses import dataclass
from optical_flow.bounding_box import BoundingBox
from optical_flow.ess import FuzzyBoundingBox
from optical_flow.ess import ess_search
from optical_flow.ess import get_largest_bounding_box

@dataclass
class DroTrackBBOXStats:
    scale : float  # h / frame.h
    delta : tuple[float, float] 

def points_clean_up(points : np.ndarray, std = 3) -> np.ndarray:

    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])

    std_x = np.std(points[:, 0])
    std_y = np.std(points[:, 1])

    subset_x = np.logical_and(points[:, 0] >= mean_x - 3 * std_x, points[:, 0] <= mean_x + 3 * std_x)
    subset_y = np.logical_and(points[:, 1] >= mean_y - 3 * std_y, points[:, 1] <= mean_y + 3 * std_y)

    return points[np.logical_and(subset_x, subset_y), :]

def points_to_bbox(points : np.ndarray) -> BoundingBox:
    # Multiple points case
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)

def drotrack_bbox_step(frame : np.ndarray, prev_bbox : BoundingBox, points : np.ndarray, stats : DroTrackBBOXStats) -> tuple[tuple[int, int], DroTrackBBOXStats]:
    if points.shape[0] == 0: return None, None

    points = points_clean_up(points)
    computed_bbox = points_to_bbox(points)

    curr_scale = prev_bbox.h / frame.shape[0]
    scale = curr_scale / stats.scale
    
    return (computed_bbox.cx + scale * stats.delta[0], computed_bbox.cy + scale * stats.delta[1]), stats

def drotrack_bbox_init(frame : np.ndarray, points : np.ndarray, bbox : BoundingBox) -> DroTrackBBOXStats:
    points = points_clean_up(points)
    computed_bbox = points_to_bbox(points)

    return DroTrackBBOXStats(bbox.h / frame.shape[0], (bbox.cx - computed_bbox.cx, bbox.cy - computed_bbox.cy))


def subset_points(points : np.ndarray, bbox : BoundingBox):
    subset_x = np.logical_and(points[:, 0] >= bbox.x, points[:, 0] <= bbox.x + bbox.w)
    subset_y = np.logical_and(points[:, 1] >= bbox.y, points[:, 1] <= bbox.y + bbox.h)
    subset = np.logical_and(subset_x, subset_y)

    return points[subset, ...]

def compute_bbox(points : np.ndarray, 
                 search_window : BoundingBox,
                 prev_bbox : BoundingBox = None, 
                 weight_pts : float = 1.0, 
                 weight_delta : float = 1.0) -> Tuple[BoundingBox, float]:

    def ess_search_function(bbox : FuzzyBoundingBox) -> float:
        
        large = get_largest_bounding_box(bbox)
        
        points_in = subset_points(points, large)
        points_sum = len(points_in)

        # FIT TIGHTER
        # left
        diff_l = np.abs(points[:, 0] -  bbox.l.mid_point)
        delta_l = np.clip(np.min(diff_l) - bbox.l.span/2, 0, np.inf)

        # right
        diff_r = np.abs(points[:, 0] -  bbox.r.mid_point)
        delta_r = np.clip(np.min(diff_r) - bbox.r.span/2, 0, np.inf)

        # bottom
        diff_b = np.abs(points[:, 1] -  bbox.b.mid_point)
        delta_b = np.clip(np.min(diff_b) - bbox.b.span/2, 0, np.inf)

        # top
        diff_t = np.abs(points[:, 1] -  bbox.t.mid_point)
        delta_t = np.clip(np.min(diff_t) - bbox.t.span/2, 0, np.inf)

        return weight_pts * points_sum - weight_delta * (delta_l + delta_r + delta_b + delta_t)

    return ess_search(search_window, ess_search_function)