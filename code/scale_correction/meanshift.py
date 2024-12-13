import numpy as np
import cv2

# cap = cv2.VideoCapture('slow.flv')
# sequence_path = "sequences/uav0000099_02520_s/"
# annotation = "annotations/uav0000099_02520_s.txt"
sequence_path = "sequences/uav0000169_00000_s/"
annotation = "annotations/uav0000169_00000_s.txt"

# Read all images in the sequence
import os
images = []
for filename in os.listdir(sequence_path):
    img = cv2.imread(os.path.join(sequence_path, filename))
    if img is not None:
        images.append(img)

print("Number of images in sequence: ", len(images))
# cv.imshow("Frame", images[0])
# cv.waitKey(0)
# exit()
# Read all annotations
with open(annotation, "r") as f:
    annotations = f.readlines()

# Extract the first annotation

x, y, w, h = map(int, annotations[0].split(","))
frame = images[0]
# x, y, w, h = 625, 375, 75, 125

# Draw the rectangle on the first frame
cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
cv2.imshow("Frame", frame)
cv2.waitKey(0)


# take first frame of the video
# ret,frame = cap.read()

# setup initial location of window
# r,h,c,w = 250,90,400,125  # simply hardcoded the values
# track_window = (c,r,w,h)
track_window = (x, y, w, h)

# set up the ROI for tracking
# roi = frame[r:r+h, c:c+w]
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# mask = cv2.inRange(hsv_roi, np.array((0., 30., 30.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT*5, 10, 1 )

# while(1):
for frame in images:

    # ret ,frame = cap.read()

    if True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()