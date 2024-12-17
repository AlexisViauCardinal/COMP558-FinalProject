import cv2 as cv
import numpy as np
import os
import time
from utils import test_utils as T


class SIFTTracker:
    def __init__(self, first_frame, x, y, w, h, first_frame_features_only=True):
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

if __name__ == "__main__":
    # Load sequence
    sequence_path = "datasets/VisDrone2019-SOT-train/sequences/uav0000085_00000_s/"
    images = []
    for filename in os.listdir(sequence_path):
        img = cv.imread(os.path.join(sequence_path, filename))
        if img is not None:
            images.append(img)

    # Load annotations
    annotation = "datasets/VisDrone2019-SOT-train/annotations/uav0000085_00000_s.txt"
    with open(annotation, "r") as f:
        annotations = f.readlines()

    # Extract the first annotation
    x, y, w, h = map(int, annotations[0].split(","))

    # Initialize the SIFT tracker
    frame = images[0]
    sift_tracker = SIFTTracker(frame, x, y, w, h)
    test_frames = [1, 44, 300, 305, 310]

    # Track the object in the sequence
    for i, frame in enumerate(images):
        if i+1 in test_frames:
            frame_copy = frame.copy()
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, str(i+1), (x, w), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv.addText(frame, "Predicted", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw ground truth bounding box
            x_gt, y_gt, w_gt, h_gt = map(int, annotations[i].split(","))
            # cv.rectangle(frame, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 0, 255), 2)
            # cv.addText(frame, "Ground Truth", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv.imshow("Frame", frame)
            cv.rectangle(frame_copy, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 0, 255), 2)
            cv.imshow("Frame", frame)
            cv.waitKey(0)
            print(f"Frame {i+1}")   
            # cv.waitKey(0)
            # cv.imwrite(f"sift_tracker_{i+1}.jpg", frame)
            # cv.imwrite(f"sift_tracker_{i+1}_gt.jpg", frame_copy)
            if i+1 == test_frames[-1]:
                break
        # cv.destroyAllWindows()
        bbox = sift_tracker.track(frame)
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        # Draw the bounding box on the frame


