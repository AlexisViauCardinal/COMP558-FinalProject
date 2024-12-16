from dataclasses import dataclass
import numpy as np

@dataclass
class BoundingBox:
    x : int
    y : int
    w : int
    h : int

    def __post_init__(self):
        self.cx = int(self.x + self.w/2)
        self.cy = int(self.y + self.h/2)

@dataclass
class DroTrackBBOXStats:
    scale : float  # h / frame.h
    delta : tuple[float, float] 
    
def center_to_bbox(cx : float, cy : float, w : float, h : float) -> BoundingBox:
    return BoundingBox(int(cx - w / 2), int(cy - h/2), int(w), int(h))

def __points_clean_up(points : np.ndarray, std = 3) -> np.ndarray:
    mean_x = np.mean(points[:, 0, 0])
    mean_y = np.mean(points[:, 0, 1])

    std_x = np.std(points[:, 0, 0])
    std_y = np.std(points[:, 0, 1])

    subset_x = np.logical_and(points[:, 0, 0] > mean_x - 3 * std_x, points[:, 0, 0] < mean_x + 3 * std_x)
    subset_y = np.logical_and(points[:, 0, 1] > mean_y - 3 * std_y, points[:, 0, 1] < mean_y + 3 * std_y)

    return points[np.logical_and(subset_x, subset_y), :, :]

def __points_to_bbox(points : np.ndarray) -> BoundingBox:
    x_min = np.min(points[:, 0, 0])
    x_max = np.max(points[:, 0, 0])

    y_min = np.min(points[:, 0, 1])
    y_max = np.max(points[:, 0, 1])

    return BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)

def drotrack_bbox_step(frame : np.ndarray, prev_bbox : BoundingBox, points : np.ndarray, stats : DroTrackBBOXStats) -> tuple[tuple[int, int], DroTrackBBOXStats]:
    
    points = __points_clean_up(points)
    computed_bbox = __points_to_bbox(points)

    curr_scale = prev_bbox.h / frame.shape[0]
    scale = curr_scale / stats.scale
    
    return (computed_bbox.x - scale * stats.delta[0], computed_bbox.y - scale * stats.delta[1]), stats

def drotrack_bbox_init(frame : np.ndarray, points : np.ndarray, bbox : BoundingBox) -> DroTrackBBOXStats:
    points = __points_clean_up(points)

    computed_bbox = __points_to_bbox(points)

    return DroTrackBBOXStats(bbox.h / frame.shape[0], (bbox.x - computed_bbox.cx, bbox.y - computed_bbox.cy))

def scale_bounding_box(bbox : BoundingBox, scale : float) -> BoundingBox:
    return BoundingBox(int(bbox.x * scale), int(bbox.y * scale), int(bbox.w * scale), int(bbox.h * scale))

def expand_bounding_box(bbox : BoundingBox, scale : float) -> BoundingBox:
    return center_to_bbox(bbox.cx, bbox.cy, scale * bbox.w, scale * bbox.h)

def bound_bounding_box(bbox : BoundingBox, min, max) -> BoundingBox:
    x = np.clip(bbox.x, min[0], max[0])
    w = np.clip(bbox.x + bbox.w, min[0], max[0]) - x

    y = np.clip(bbox.y, min[1], max[1])
    h = np.clip(bbox.y + bbox.h, min[1], max[1]) - y

    return BoundingBox(x, y, w, h)