
import csv
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2
import csv

@dataclass
class BoundingBox:
    x: int  # x-coordinate of the top-left corner
    y: int  # y-coordinate of the top-left corner
    h: int  # height of the bounding box
    w: int  # width of the bounding box

    def __post_init__(self):
        self.cx = int(self.x + self.w/2)
        self.cy = int(self.y + self.h/2)

def list_to_Bbox_obj(bbox_list):
    bbox_obj = BoundingBox(int (bbox_list[0]), int (bbox_list[1]), int (bbox_list[2]), int (bbox_list[3]))
    return bbox_obj

def Bbox_obj_to_list(bbox_obj):
    bbox_list = [bbox_obj.x, bbox_obj.y, bbox_obj.h, bbox_obj.w]
    return bbox_list

class ErrorTracker:
    def __init__(self, ground_truth_csv):
        self.ground_truth, self.num_frames = self._load_ground_truth(ground_truth_csv)
        self.ious = np.zeros(self.num_frames)
        self.center_errors = np.zeros(self.num_frames)
        self.iteration = 0  # Track the current iteration
    
    def _load_ground_truth(self, csv_file):
        """
        Load ground truth bounding boxes from CSV file. 
        - Assumes each row in the csv corresponds to (x_coord, y_coord, width, height)
        - Assumes no header in the csv
        """
        num_entries = 0
        ground_truth = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                num_entries += 1
                x, y, w, h = map(int, row)
                ground_truth.append(BoundingBox(x, y, h, w))
        return ground_truth, num_entries
    
    def compute_iou(self, gt: BoundingBox, pred: BoundingBox) -> float:
        """ Compute Intersection over Union (IoU) between ground truth and predicted bounding boxes. """
        # Get the coordinates of the intersection area
        x1 = max(gt.x, pred.x)
        y1 = max(gt.y, pred.y)
        x2 = min(gt.x + gt.w, pred.x + pred.w)
        y2 = min(gt.y + gt.h, pred.y + pred.h)
        
        # Calculate area of intersection
        intersection_w = max(0, x2 - x1)
        intersection_h = max(0, y2 - y1)
        intersection_area = intersection_w * intersection_h
        
        # Calculate area of both bounding boxes
        gt_area = gt.w * gt.h
        pred_area = pred.w * pred.h
        
        # Calculate union area
        union_area = gt_area + pred_area - intersection_area
        
        # IoU formula
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def compute_center_error(self, gt: BoundingBox, pred: BoundingBox) -> float:
        """ Compute center error between ground truth and predicted box """
        return math.sqrt((gt.x - pred.x)**2 + (gt.y - pred.y)**2)
    
    def update(self, predicted_bbox: BoundingBox):
        """ Update the error list with the IoU score for the current iteration. """
        if self.iteration < len(self.ground_truth):
            gt_bbox = self.ground_truth[self.iteration]
            iou = self.compute_iou(gt_bbox, predicted_bbox)
            err = self.compute_center_error(gt_bbox, predicted_bbox)
            self.ious[self.iteration] = iou
            self.center_errors[self.iteration] = err 
            self.iteration += 1
            print("Iteration: ", self.iteration, ", center error: ", err, ", IoU: ", iou)
        else:
            print("All ground truth bounding boxes have been processed.")
    
    def get_ious(self):
        """ Return the list of IoU errors. """
        return self.ious
    
    def get_AOS(self): 
        """ Return the average overlap score. """
        return np.average(self.ious)

    def get_tracking_length(self, threshold):
        """ Return the length (number of frames) of successful tracking based on the specified threshold """
        # get the index of the first entry greater than threshold in self.center_errors
        successful_frames = 0
        for error in self.center_errors:
            if error <= threshold:
                successful_frames += 1
            else:
                break  
        return successful_frames

def write_tracking_info_to_csv(tracking_data, output_file, error_threshold):
    """
    Write tracking data to a CSV file, arranged with columns for each video,
    divided into IoUs and Center Errors, with AOS and Tracking Length at the top.
    
    Parameters:
    - tracking_data (dict): Dictionary where keys are video names and values are TrackingError objects.
    - output_file (str): Path to save the CSV file.
    - error_threshold (float): Threshold for computing successful tracking length.
    """
    # Prepare the data for writing
    video_names = list(tracking_data.keys())
    max_entries = max(len(tracker.get_ious()) for tracker in tracking_data.values())

    # Open the file in write mode
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row with video names, AOS, and tracking length
        header_row = []
        for video_name in video_names:
            tracker = tracking_data[video_name]
            aos = tracker.get_AOS()
            tracking_length = tracker.get_tracking_length(error_threshold)
            header_row.extend([f"{video_name} (AOS={aos}, TL={tracking_length})", ""])
        writer.writerow(header_row)
        
        # Write the sub-header for "IoUs" and "Center Errors"
        sub_header_row = []
        for _ in video_names:
            sub_header_row.extend(["IoUs", "Center Errors"])
        writer.writerow(sub_header_row)
        
        # Write the IoUs and Center Errors for each video
        for i in range(max_entries):
            row = []
            for video_name in video_names:
                tracker = tracking_data[video_name]
                ious = tracker.get_ious()
                center_errors = tracker.center_errors
                
                iou = ious[i] if i < len(ious) else ""
                center_error = center_errors[i] if i < len(center_errors) else ""
                row.extend([iou, center_error])
            writer.writerow(row)
    
    print(f"Tracking data saved to {output_file}")

def create_video_writer(output_path, frame_size, fps):
    """
    Initialize a video writer to save frames as a video.

    Parameters:
    - output_path (str): Path to save the video.
    - frame_size (tuple): Size of the frames (width, height).
    - fps (int): Frames per second.

    Returns:
    - video_writer (cv2.VideoWriter): The initialized video writer.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4, 'XVID' for .avi
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return video_writer