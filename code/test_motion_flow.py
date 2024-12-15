import numpy as np
import cv2 as cv
import os

folder_path = "/workspaces/python-opencv/resources/VisDrone2019-SOT-train/sequences/uav0000169_00000_s/"
out_path = "/workspaces/python-opencv/repo/out/converted.mp4"

fps = 30

pictures = sorted(os.listdir(folder_path))

frame1 = cv.imread(folder_path + pictures[0])

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

fourcc = cv.VideoWriter_fourcc(*'mp4v')
video_out = cv.VideoWriter(out_path, fourcc, fps, frame1.shape[0:2][::-1]) 

for picture in pictures[1:]:
    frame2 = cv.imread(folder_path + picture)

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    video_out.write(bgr)
    k = cv.waitKey(30) & 0xff
    
    prvs = next

video_out.release()
cv.destroyAllWindows()