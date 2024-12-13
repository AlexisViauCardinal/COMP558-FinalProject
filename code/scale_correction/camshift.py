from scale_correction import ScaleCorrection
import numpy as np
import cv2 as cv

class CamShift(ScaleCorrection):
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
        # mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        h, s, v = self.find_dom_color(subwindow) 
        lower_bound = np.array([max(h - 10, 0), max(s - 50, 0), max(v - 50, 0)])
        upper_bound = np.array([min(h + 10, 180), min(s + 50, 255), min(v + 50, 255)])
        mask = cv.inRange(hsv_roi, lower_bound, upper_bound)
        self.roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0,180])
        cv.normalize(self.roi_hist, self.roi_hist, 0, 255, cv.NORM_MINMAX)
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    def find_dom_color(self, subwindow):
        hsv_subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2HSV)
        pixels = hsv_subwindow.reshape((-1, 3))
        pixels = np.float32(pixels)
        k = 3
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        label_counts = np.bincount(labels.flatten())
        dominant_index = np.argmax(label_counts)
        dominant_color = centers[dominant_index]
        return dominant_color

    def correct_scale(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        self.w_prev = self.track_window[2]
        self.h_prev = self.track_window[3]
        ret, self.track_window = cv.CamShift(dst, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        new_scale = 0.5 * (w/self.w_prev + h/self.h_prev) # Take the average of the scale in x and y
        return new_scale
    

# Test the CamShift class
if __name__=="__main__":
    video_path = "videos/jog.mp4"
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    x, y, w, h = 625, 375, 75, 125
    camshift = CamShift(frame, x, y, w, h)
    # Show frame and print new scale
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        new_scale = camshift.correct_scale(frame)
        # Draw the rectangle
        x, y, w, h = camshift.track_window
        cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        print(new_scale)
        # Show frame at half size
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("Frame", frame)
        
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # break
