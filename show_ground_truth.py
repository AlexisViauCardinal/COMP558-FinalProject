import cv2 
import numpy as np
import os

sequence_path = f"sequences/uav0000169_00000_s"
annotation_path = f"annotations/uav0000169_00000_s.txt"

# Load the sequence
sequence = []
# Load all images in folder
for filename in os.listdir(sequence_path):
    img = cv2.imread(os.path.join(sequence_path, filename))
    sequence.append(img)

# Load the annotations
annotations = []
with open(annotation_path, 'r') as file:
    for line in file:
        annotations.append(list(map(int, line.split(','))))

# Draw first annotation on first image
# img = sequence[0]
# annotation = annotations[0]
# x1, y1, w, h = annotation
# cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

# For the first 100 frames, draw bounding box 
for i in range(1, 1000, 100):
    img = sequence[i]
    annotation = annotations[i]
    x1, y1, w, h = annotation
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
    cv2.imshow('frame', img)
    cv2.waitKey(0)


