import cv2 as cv
import numpy as np
import os
import time
from utils import test_utils as T


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
        # bbox = [int(x_new), int(y_new), int(x_new + w), int(y_new + h)]
        # bbox = T.list_to_Bbox_obj([int(x_new), int(y_new), int(x_new + w), int(y_new + h)])
        # bbox = [int(x_new), int(y_new), h, w]
        bbox = T.list_to_Bbox_obj([int(x_new), int(y_new), h, w])
        # center = bbox_helper.get_bbox_center(bbox)
        # return bbox, center, extime
        return bbox

        

# CODE FOR REPORT IMAGE

# if __name__ == "__main__":
#     # Load frame 1 and frame 5
#     frame1 = cv.imread("sequences/uav0000085_00000_s/img0000001.jpg")
#     frame2 = cv.imread("sequences/uav0000085_00000_s/img0000005.jpg")

#     # Get the first annotation
#     with open("annotations/uav0000085_00000_s.txt", "r") as f:
#         annotations = f.readlines()
#     x, y, w, h = map(int, annotations[0].split(","))

#     # Create a copy of the first frame and draw the bounding box
#     frame1_with_box = frame1.copy()
#     cv.rectangle(frame1_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle

#     # Detect keypoints in bounding box
#     sift = cv.SIFT_create()
#     roi = frame1[y:y+h, x:x+w]
#     kp1, des1 = sift.detectAndCompute(roi, None)

#     # Detect keypoints and descriptors in the second frame
#     kp2, des2 = sift.detectAndCompute(frame2, None)

#     # Match descriptors
#     bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
#     matches = bf.match(des1, des2)

#     # Sort matches
#     matches = sorted(matches, key=lambda match: match.distance)
#     num_matches = min(10, len(matches))  # Take top 10

#     # Shift keypoints to frame1 coordinates
#     for kp in kp1:
#         kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

#     # Draw matches 
#     matches_image = cv.drawMatches(
#         frame1_with_box, kp1,  
#         frame2, kp2,          
#         matches[:num_matches], 
#         None,                 
#         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#     matches_image = cv.resize(matches_image, (0, 0), fx=0.4, fy=0.4)
#     # cv.imshow("Bounding Box and Matches", matches_image)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
#     cv.imwrite("SIFT_matching.jpg", matches_image)