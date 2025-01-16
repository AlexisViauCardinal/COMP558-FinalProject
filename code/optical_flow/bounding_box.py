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

    
def center_to_bbox(cx : float, cy : float, w : float, h : float) -> BoundingBox:
    return BoundingBox(int(cx - w / 2), int(cy - h/2), int(w), int(h))

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