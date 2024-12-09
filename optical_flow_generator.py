import numpy as np
import cv2 as cv

# Path to input video
video_path = "videos/jog.mp4"
cap = cv.VideoCapture(video_path)

# Bounding box coordinates
x, y, w, h = 625, 375, 75, 125
x1, x2, y1, y2 = x, x + w, y, y + h



# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors for optical flow tracks
color = np.random.randint(0, 255, (100, 3))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    exit()

# Convert to grayscale and detect initial corners
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Filter points within the bounding box
p0 = p0[(p0[:, 0, 0] >= x1) & (p0[:, 0, 0] <= x2) & (p0[:, 0, 1] >= y1) & (p0[:, 0, 1] <= y2)]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Get frame width, height, and FPS for saving the output video
frame_height, frame_width = old_frame.shape[:2]
fps = cap.get(cv.CAP_PROP_FPS) * 3

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 files
out = cv.VideoWriter('flow_output.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Convert frame to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    # Overlay the mask on the frame
    img = cv.add(frame, mask)

    # Write the frame to the output file
    out.write(img)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()
