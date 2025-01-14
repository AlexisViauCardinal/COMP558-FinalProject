# %%
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# %%
sys.path.append("/home/alexis/Documents/COMP558/COMP558-FinalProject/code/")
# sys.path.append("/workspaces/python-opencv/repo/code/")


# %%
from optical_flow.bounding_box import BoundingBox
from feature_description.orb_descriptor import ORBDescriptor
from optical_flow.gu import Gu

# %%
video_name = "/home/alexis/Documents/master/vision/repo/libfreenect/wrappers/python/out/VIDEO-20250113-190349.mp4"
depth_name = "/home/alexis/Documents/master/vision/repo/libfreenect/wrappers/python/out/DEPTH-20250113-190349.mp4"

output_name = "/home/alexis/Documents/COMP558/COMP558-FinalProject/out/gu_py_box_steps.mp4"

# %%
x, y, w, h = 275, 200, 110, 85

orb_params = {"params": {"nfeatures" : 1000, "edgeThreshold" : 15, "patchSize" : 10}}

# %%
initial_bbox = BoundingBox(x, y, w, h)
# initial_bbox = expand_bounding_box(initial_bbox, 1.25)

feature_descriptor = ORBDescriptor(**orb_params)

# %%
cap_video = cv.VideoCapture(video_name)

ret_video, frame_video = cap_video.read()

# %%
gu_params = {"_lambda" : 4/5, "frame_buffer" : 60, "gamma" : 0.1, "gamma_fit" : 0.5, "gamma_area" : 0.1, "gamma_drift" : 0.5}
gu = Gu(frame_video, initial_bbox, feature_descriptor, **gu_params)

# %%
fps = cap_video.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
video_writer = cv.VideoWriter(output_name, fourcc, fps, frame_video.shape[:-1][::-1])

# %%
length = int(cap_video.get(cv.CAP_PROP_FRAME_COUNT))

frequency = 3

# while ret_video and ret_depth:
for i in tqdm(range(np.clip(length, 0, 1000))):

    if not ret_video:
        break
    
    if i % frequency == 0:

        bbox, _ = gu.track_frame(frame_video)

        img2 = cv.rectangle(frame_video, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), 255, 2)

        video_writer.write(img2)

    ret_video, frame_video = cap_video.read()

# %%
video_writer.release()


