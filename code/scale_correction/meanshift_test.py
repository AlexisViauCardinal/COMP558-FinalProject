import numpy as np
import cv2 as cv

video_path = "videos/jog.mp4"

cap = cv.VideoCapture(video_path)

# take first frame of the video
ret, frame = cap.read()

# setup initial location of window
x, y, w, h = 625, 375, 75, 125  # simply hardcoded the values
track_window = (x, y, w, h)

# Step 1: Extract ROI
roi = frame[y:y+h, x:x+w]

# Define the central subwindow (half the size of the ROI, centered in the middle)
subwindow_w, subwindow_h = w // 2, h // 2
subwindow_x = w // 4  # Center horizontally within ROI
subwindow_y = h // 4  # Center vertically within ROI
subwindow = roi[subwindow_y:subwindow_y + subwindow_h, subwindow_x:subwindow_x + subwindow_w]

# Convert the subwindow to HSV
hsv_subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2HSV)

# Step 2: Reshape the subwindow for k-means clustering
pixels = hsv_subwindow.reshape((-1, 3))
pixels = np.float32(pixels)

# Step 3: Apply k-means clustering
k = 3  # Use a small number of clusters for simplicity
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Step 4: Find the most dominant cluster
label_counts = np.bincount(labels.flatten())  # Count the number of pixels in each cluster
dominant_index = np.argmax(label_counts)      # Index of the most frequent cluster
dominant_color = centers[dominant_index]      # HSV value of the dominant color

# Step 5: Define an HSV range around the dominant color
h, s, v = dominant_color
lower_bound = np.array([max(h - 10, 0), max(s - 50, 0), max(v - 50, 0)])
upper_bound = np.array([min(h + 10, 180), min(s + 50, 255), min(v + 50, 255)])

# Step 6: Create the mask for the dominant color
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, lower_bound, upper_bound)

# Use the mask to calculate the histogram
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv.imshow('img2', img2)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
