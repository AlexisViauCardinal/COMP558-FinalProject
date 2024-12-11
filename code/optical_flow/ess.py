import numpy as np
from queue import PriorityQueue
from optical_flow.bounding_box import BoundingBox
from dataclasses import dataclass
from typing import Callable

@dataclass
class Interval:
    low : int   # x >= low
    high : int  # x < high

@dataclass
class FuzzyBoundingBox:
    t : Interval
    b : Interval
    l : Interval
    r : Interval

def interval_span(i : Interval) -> int:
    return np.abs(i.high - i.low)

def split_interval(interval : Interval) -> tuple[Interval, Interval]:
    
    mid = (interval.low + interval.high) / 2
    mid = int(mid)

    if mid == interval.low or mid + 1 >= interval.high:
        # returns null if only one interval is feasible
        return (Interval(mid, mid + 1), None)
    
    return Interval(interval.low, mid), Interval(mid + 1, interval.high)

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
            return FuzzyBoundingBox(a, fuzzy.b, fuzzy.l, fuzzy.r), FuzzyBoundingBox(b, fuzzy.b, fuzzy.l, fuzzy.r)
        case 1:
            a, b = split_interval(fuzzy.b)
            return FuzzyBoundingBox(fuzzy.t, a, fuzzy.l, fuzzy.r), FuzzyBoundingBox(fuzzy.t, b, fuzzy.l, fuzzy.r)
        case 2:
            a, b = split_interval(fuzzy.l)
            return FuzzyBoundingBox(fuzzy.t, fuzzy.b, a, fuzzy.r), FuzzyBoundingBox(fuzzy.t, fuzzy.b, b, fuzzy.r)
        case 3:
            a, b = split_interval(fuzzy.r)
            return FuzzyBoundingBox(fuzzy.t, fuzzy.b, fuzzy.l, a), FuzzyBoundingBox(fuzzy.t, fuzzy.b, fuzzy.l, b)
        case _:
            raise ValueError("Internal processing error")

def is_interval_single(interval : Interval) -> bool:
    return interval.low == interval.high or interval.low + 1 == interval.high 

def is_fuzzy_box_single(box : FuzzyBoundingBox) -> bool:
    return is_interval_single(box.t) and is_interval_single(box.b) and is_interval_single(box.l) and is_interval_single(box.r)

def bounding_box_to_fuzzy(bounding_box : BoundingBox) -> FuzzyBoundingBox:
    ## TODO CHANGE TO BE COHERENT WITH DIAGRAM
    v_low = bounding_box.y
    v_high = v_low + bounding_box.h
    v_interval = Interval(v_low, v_high)

    h_low = bounding_box.x
    h_high = h_low + bounding_box.w
    h_interval = Interval(h_low, h_high)

    return FuzzyBoundingBox(v_interval, v_interval, h_interval, h_interval)

def fuzzy_to_bounding_box(fuzzy : FuzzyBoundingBox) -> BoundingBox:
    
    if not is_fuzzy_box_single(fuzzy) : 
        raise ValueError("Bounding box should not be fuzzy when converting")
    
    v_low = fuzzy.b.low
    v_high = fuzzy.t.low

    h_low = fuzzy.l.low
    h_high = fuzzy.r.high
    
    return BoundingBox(h_low, v_low, h_high - h_low, v_high - v_low)



def search(self, max_box : BoundingBox, evaluation_function : Callable[[BoundingBox], float]):
    '''
        Perform the Efficient Subwindow Search. 
        Note that the image properties/statistics should be implicit to the evaluation function

        max_box : maximum range to perform the search (BoundingBox)
        evaluation_function : function approximating the performance of the classifier

        Returns the best bounding box.
    '''


    current_box = bounding_box_to_fuzzy(max_box)

    queue = PriorityQueue()
    queue.put((evaluation_function(current_box), current_box))

    while not is_fuzzy_box_single(current_box):
        
        a, b = split_fuzzy(current_box)


        pass

    pass