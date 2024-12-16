import numpy as np
from queue import PriorityQueue
from optical_flow.bounding_box import BoundingBox
from dataclasses import dataclass
from typing import Callable
from itertools import count

@dataclass
class Interval:
    low : int   # x >= low
    high : int  # x < high

    def __post_init__(self):
        if self.low > self.high:
            print("Warning Interval")
            low, high = high, low

@dataclass
class FuzzyBoundingBox:
    # Assume t >= b, r >= l
    t : Interval
    b : Interval
    l : Interval
    r : Interval

def interval_span(interval : Interval) -> int:
    return np.abs(interval.high - interval.low)

def split_interval(interval : Interval) -> tuple[Interval, Interval]:
    
    if interval_span(interval) <= 1:
        return interval, None

    mid = (interval.low + interval.high) / 2
    mid = int(mid)
    
    return Interval(interval.low, mid), Interval(mid, interval.high)

def interval_overlap(i1 : Interval, i2 : Interval) -> bool:
    v1 = sorted([i1.low, i1.high])
    v2 = sorted([i2.low, i2.high])
    return (v1[1] >= v2[0] and v1[0] <= v2[1]) or (v2[1] >= v1[0] and v2[0] <= v1[1])

def guard_fuzzy(fuzzy : FuzzyBoundingBox) -> FuzzyBoundingBox:
    if fuzzy.b.low > fuzzy.t.high:
        return None
    if fuzzy.l.low > fuzzy.r.high:
        return None
    return fuzzy 

def split_fuzzy(fuzzy : FuzzyBoundingBox) -> tuple[FuzzyBoundingBox, FuzzyBoundingBox]:
    
    t_span = interval_span(fuzzy.t)
    b_span = interval_span(fuzzy.b)
    l_span = interval_span(fuzzy.l)
    r_span = interval_span(fuzzy.r)

    if np.max((t_span, b_span, l_span, r_span)) <= 1:
        return fuzzy, None

    max_id = np.argmax((t_span, b_span, l_span, r_span))

    match int(max_id):
        case 0:
            a, b = split_interval(fuzzy.t)
            return guard_fuzzy(FuzzyBoundingBox(a, fuzzy.b, fuzzy.l, fuzzy.r)), guard_fuzzy(FuzzyBoundingBox(b, fuzzy.b, fuzzy.l, fuzzy.r))
        case 1:
            a, b = split_interval(fuzzy.b)
            return guard_fuzzy(FuzzyBoundingBox(fuzzy.t, a, fuzzy.l, fuzzy.r)), guard_fuzzy(FuzzyBoundingBox(fuzzy.t, b, fuzzy.l, fuzzy.r))
        case 2:
            a, b = split_interval(fuzzy.l)
            return guard_fuzzy(FuzzyBoundingBox(fuzzy.t, fuzzy.b, a, fuzzy.r)), guard_fuzzy(FuzzyBoundingBox(fuzzy.t, fuzzy.b, b, fuzzy.r))
        case 3:
            a, b = split_interval(fuzzy.r)
            return guard_fuzzy(FuzzyBoundingBox(fuzzy.t, fuzzy.b, fuzzy.l, a)), guard_fuzzy(FuzzyBoundingBox(fuzzy.t, fuzzy.b, fuzzy.l, b))
        case _:
            raise ValueError("Internal processing error")

def is_interval_single(interval : Interval) -> bool:
    return interval_span(interval) <= 1

def is_fuzzy_box_single(box : FuzzyBoundingBox) -> bool:
    return is_interval_single(box.t) and is_interval_single(box.b) and is_interval_single(box.l) and is_interval_single(box.r)

def bounding_box_to_fuzzy(bounding_box : BoundingBox) -> FuzzyBoundingBox:
    ## TODO CHANGE TO BE COHERENT WITH DIAGRAM
    v_low = bounding_box.y
    v_high = v_low + bounding_box.h//2
    v_interval_b = Interval(v_low, v_high)
    v_interval_t = Interval(v_high, v_high + bounding_box.h//2)

    h_low = bounding_box.x
    h_high = h_low + bounding_box.w//2
    h_interval_l = Interval(h_low, h_high)
    h_interval_r = Interval(h_low, h_high + bounding_box.w//2)

    return FuzzyBoundingBox(v_interval_t, v_interval_b, h_interval_l, h_interval_r)

def fuzzy_to_bounding_box(fuzzy : FuzzyBoundingBox) -> BoundingBox:
    
    if not is_fuzzy_box_single(fuzzy) : 
        raise ValueError("Bounding box should not be fuzzy when converting")
    
    v_low = fuzzy.b.low
    v_high = fuzzy.t.high

    h_low = fuzzy.l.low
    h_high = fuzzy.r.high
    
    return get_largest_bounding_box(fuzzy) #BoundingBox(h_low, v_low, h_high - h_low, v_high - v_low)

def get_smallest_bounding_box(fuzzy : FuzzyBoundingBox) -> BoundingBox:

    if interval_overlap(fuzzy.b, fuzzy.t) or interval_overlap(fuzzy.l, fuzzy.r) : 
        return BoundingBox(0, 0, 0, 0)
    
    v = sorted([fuzzy.b.low, fuzzy.b.high, fuzzy.t.low, fuzzy.t.high])
    h = sorted([fuzzy.l.low, fuzzy.l.high, fuzzy.r.low, fuzzy.r.high])
    
    return BoundingBox(h[1], v[1], h[2] - h[1], v[2] - v[1]) 

def get_largest_bounding_box(fuzzy : FuzzyBoundingBox) -> BoundingBox:
    v = sorted([fuzzy.b.low, fuzzy.b.high, fuzzy.t.low, fuzzy.t.high])
    h = sorted([fuzzy.l.low, fuzzy.l.high, fuzzy.r.low, fuzzy.r.high])

    return BoundingBox(h[0], v[0], h[3] - h[0], v[3] - v[0]) 


def ess_search(max_box : BoundingBox, evaluation_function : Callable[[FuzzyBoundingBox], float]) -> tuple[BoundingBox, float]:
    '''
        Perform the Efficient Subwindow Search. 
        Note that the image properties/statistics should be implicit to the evaluation function

        max_box : maximum range to perform the search (BoundingBox)
        evaluation_function : function approximating the performance of the classifier

        Returns the best bounding box and score
    '''

    unique = count()

    current_box = bounding_box_to_fuzzy(max_box)

    queue = PriorityQueue()
    queue.put((-evaluation_function(current_box), next(unique), current_box))

    while not is_fuzzy_box_single(current_box):
        
        score, _, current_box = queue.get()

        a, b = split_fuzzy(current_box)

        if a is not None:
            queue.put((-evaluation_function(a), next(unique), a))
        if b is not None:
            queue.put((-evaluation_function(b), next(unique), b))
       
    return fuzzy_to_bounding_box(current_box), score