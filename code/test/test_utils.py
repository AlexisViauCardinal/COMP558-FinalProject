
import csv
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x: int  # x-coordinate of the top-left corner
    y: int  # y-coordinate of the top-left corner
    h: int  # height of the bounding box
    w: int  # width of the bounding box

class TrackingError:
    def __init__(self, ground_truth_csv):
        self.ground_truth = self._load_ground_truth(ground_truth_csv)
        self.errors = []  # List to store errors at each iteration
        self.iteration = 0  # Track the current iteration
    
    def _load_ground_truth(self, csv_file):
        """ 
        Load ground truth bounding boxes from CSV file. 
        - Assumes each row in the csv corresponds to (x_coord, y_coord, width, height)
        - Assumes no header in the csv
        """
        ground_truth = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y, w, h = map(int, row)
                ground_truth.append(BoundingBox(x, y, h, w))
        return ground_truth
    
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
    
    def update(self, predicted_bbox: BoundingBox):
        """ Update the error list with the IoU score for the current iteration. """
        if self.iteration < len(self.ground_truth):
            gt_bbox = self.ground_truth[self.iteration]
            iou = self.compute_iou(gt_bbox, predicted_bbox)
            self.errors.append(iou)
            self.iteration += 1
        else:
            print("All ground truth bounding boxes have been processed.")
    
    def get_errors(self):
        """ Return the list of IoU errors. """
        return self.errors