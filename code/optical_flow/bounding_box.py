from dataclasses import dataclass

@dataclass
class BoundingBox:
    x : int
    y : int
    w : int
    h : int