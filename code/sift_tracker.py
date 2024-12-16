import cv2 as cv
import numpy as np
import os
import time
from utils import bbox_helper


class SIFTTracker:
    def __init__(self, first_frame, x, y, w, h, first_frame_features_only=False):
        self.track_window = (x, y, w, h)
        self.w = w
        self.h = h
        self.max_x_move = 10
        self.max_y_move = 10
        self.sift = cv.SIFT_create()
        self.ff_features_only = first_frame_features_only

        # Initialize keypoints and descriptors for ROI
        roi = first_frame[y:y + h, x:x + w]
        self.kp1, self.des1 = self.sift.detectAndCompute(roi, None)

    def track(self, frame):
        start = time.time()
        x, y, w, h = self.track_window

        # Define the search window, ensuring it stays within bounds
        x_start = max(0, int(x - self.max_x_move))
        y_start = max(0, int(y - self.max_y_move))
        x_end = min(frame.shape[1], int(x + w + self.max_x_move))
        y_end = min(frame.shape[0], int(y + h + self.max_y_move))
        search_window = frame[y_start:y_end, x_start:x_end]

        # Detect keypoints and descriptors in the search window
        kp2, des2 = self.sift.detectAndCompute(search_window, None)

        # Match keypoints using BFMatcher
        bf = cv.BFMatcher()
        matches = bf.match(self.des1, des2)

        # Sort matches by distance and use the top matches
        matches = sorted(matches, key=lambda match: match.distance)
        num_matches = min(5, len(matches))  # Use up to 5 matches
        matched_points = [kp2[match.trainIdx].pt for match in matches[:num_matches]]

        # Compute the average coordinates of the matched points
        x_avg = np.mean([pt[0] for pt in matched_points])
        y_avg = np.mean([pt[1] for pt in matched_points])

        # Adjust the coordinates back to the frame's reference system
        x_new = x_start + x_avg - w // 2
        y_new = y_start + y_avg - h // 2

        # Ensure the new bounding box stays within bounds
        x_new = max(0, min(frame.shape[1] - w, int(x_new)))
        y_new = max(0, min(frame.shape[0] - h, int(y_new)))

        # Update track window
        self.track_window = (x_new, y_new, self.w, self.h)

        # Update the ROI keypoints and descriptors
        if not self.ff_features_only:
            roi = frame[int(y_new):int(y_new + h), int(x_new):int(x_new + w)]
            self.kp1, self.des1 = self.sift.detectAndCompute(roi, None)
        extime = time.time() - start
        bbox = [int(x_new), int(y_new), int(x_new + w), int(y_new + h)]
        center = bbox_helper.get_bbox_center(bbox)
        return bbox, center, extime

        



# if __name__ == "__main__":
#     # Define paths for sequence and annotations
#     sequence_path = "sequences/uav0000085_00000_s/"
#     annotation = "annotations/uav0000085_00000_s.txt"

#     # Read the first image in the sequence
#     images = []
#     for filename in sorted(os.listdir(sequence_path)):
#         img = cv.imread(os.path.join(sequence_path, filename))
#         images.append(img)

#     frame = images[0]

#     # Read annotations
#     with open(annotation, "r") as f:
#         annotations = f.readlines()

#     x, y, w, h = map(int, annotations[0].split(","))
#     # Draw the rectangle on the first frame
#     frame_copy = frame.copy()
#     cv.rectangle(frame_copy, (x, y), (x+w, y+h), 255, 2)
#     cv.imshow("Frame", frame_copy)
#     cv.waitKey(0)

#     sift_tracker = SIFTTracker(frame, x, y, w, h)
#     for frame in images:
#         track_window = sift_tracker.track(frame)
#         x, y, w, h = track_window
#         x = int(x)
#         y = int(y)
#         frame_copy = frame.copy()
#         print(x, y, w, h)
#         cv.rectangle(frame_copy, (x, y), (x+w, y+h), 255, 2)
#         frame_copy = cv.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
#         cv.imshow("Frame", frame_copy)
#         cv.waitKey(1)
#     cv.destroyAllWindows()
