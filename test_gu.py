# %%
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# %%
sys.path.append("/home/alexis/Documents/COMP558/COMP558-FinalProject/code/")

# %%
from optical_flow.bounding_box import BoundingBox
from feature_description.orb_descriptor import ORBDescriptor
from feature_description.sift_descriptor import SIFTDescriptor
from optical_flow.gu import Gu

# %%
video_name = "videos/VIDEO-20250109-160653.mp4"
depth_name = "videos/DEPTH-20250109-160653.mp4"

output_name = "out/gu2.mp4"

# %%
# x, y, w, h = 255, 280, 160, 95
x, y, w, h = 255//2, 280//2, 160//2, 95//2

orb_params = {"params": {"nfeatures" : 10000, "edgeThreshold" : 5, "patchSize" : 5}}

# %%
initial_bbox = BoundingBox(x, y, w, h)

feature_descriptor = ORBDescriptor(**orb_params)
# feature_descriptor = SIFTDescriptor()

# %%
cap_video = cv.VideoCapture(video_name)

ret_video, frame_video = cap_video.read()

# %%
# fig, ax = plt.subplots()

# ax.imshow(frame_video[:,:,::-1])

# rect = patches.Rectangle((initial_bbox.x, initial_bbox.y), initial_bbox.w, initial_bbox.h, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)

# plt.show()

# %%
scale_factor = 1/2
frame_video = cv.resize(frame_video, (0, 0), fx=scale_factor, fy=scale_factor)

print(frame_video.shape)

gu = Gu(frame_video, initial_bbox, feature_descriptor, _lambda = 4/5, frame_buffer=60, gamma=0.1)

# %%
fps = cap_video.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
video_writer = cv.VideoWriter(output_name, fourcc, fps, frame_video.shape[:-1][::-1])

# %%
length = int(cap_video.get(cv.CAP_PROP_FRAME_COUNT))

scaled = frame_video

# while ret_video and ret_depth:
for i in tqdm(range(length)):

    if not ret_video:
        break
    
    points_loc, points_desc, points_size = feature_descriptor.detect_features(scaled)
    points_loc = np.array(points_loc)

    bbox, _, img2 = gu.track_frame(scaled)

    img2 = cv.rectangle(scaled, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), 255, 2)
    
    # for j in range(points_loc.shape[0]):
        # img2 = cv.circle(img2, np.int_(points_loc[j]), 1, (0,0,255), -1)

    video_writer.write(img2)

    ret_video, frame_video = cap_video.read()
    
    if ret_video:
        scaled = cv.resize(frame_video, (0, 0), fx=scale_factor, fy=scale_factor)

# %%
video_writer.release()


