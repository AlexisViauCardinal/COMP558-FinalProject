import numpy as np
import cv2 as cv
from utils import test_utils as T

class MeanShift:
    def __init__(self, first_frame, x, y, w, h):
        self.track_window = (x, y, w, h)
        self.w_prev = w
        self.h_prev = h
        self.roi_hist = None
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        self.first_frame = first_frame
        roi = first_frame[y:y+h, x:x+w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        subwindow_w, subwindow_h = w // 2, h // 2
        subwindow_x = w // 4  # Center horizontally within ROI
        subwindow_y = h // 4  # Center vertically within ROI
        subwindow = roi[subwindow_y:subwindow_y + subwindow_h, subwindow_x:subwindow_x + subwindow_w]
        h, s, v = self.find_dom_color(subwindow) 
        lower_bound = np.array([max(h - 30, 0), max(s - 50, 0), max(v - 50, 0)], dtype=np.uint8)
        upper_bound = np.array([min(h + 30, 180), min(s + 50, 255), min(v + 50, 255)], dtype=np.uint8)
        mask = cv.inRange(hsv_roi, lower_bound, upper_bound)
        self.roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(self.roi_hist, self.roi_hist, 0, 128, cv.NORM_MINMAX)
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 0.5)

    def find_dom_color(self, subwindow):
        hsv_subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2HSV)
        pixels = hsv_subwindow.reshape((-1, 3))
        pixels = np.float32(pixels)
        k = 7
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        label_counts = np.bincount(labels.flatten())
        dominant_index = np.argmax(label_counts)
        dominant_color = centers[dominant_index]
        return dominant_color

    def track(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # Show Backprojection
        # cv.imshow("Backprojection", dst)
        # cv.waitKey(0)
        self.w_prev = self.track_window[2]
        self.h_prev = self.track_window[3]
        # ret, self.track_window = cv.CamShift(dst, self.track_window, self.term_crit)
        ret, self.track_window = cv.meanShift(dst, self.track_window, self.term_crit)
        x, y, w, h = self.track_window

        # Adjust track window dimensions
        w = int(self.w_prev)
        h = int(self.h_prev)
        cx, cy = x + w // 2, y + h // 2
        x = max(0, cx - w // 2)
        y = max(0, cy - h // 2)

        # Update the track window
        self.track_window = (x, y, w, h)
        bbox = T.list_to_Bbox_obj([int(x), int(y), h, w])
        return bbox
    
