import utils.bbox_helper as bbox_helper
from utils.logging_utils import time_stamp
import data.datasets as data
import tracker as our_tracker
import utils.test_utils as T
from feature_detection.shi_tomasi import ShiTomasiDetector
from optical_flow.lucas_kanade import LucasKanade
from feature_description.sift_descriptor import SIFTDescriptor
from segmentation.fuzzy_c_means import FuzzyCMeansSegmenter
from feature_description.ORB_descriptor import ORBDescriptor

from collections import OrderedDict
import os
import cv2
import math
import os
from tqdm import tqdm

############ Params ############

s = 1 # scaling of images (2 means scale down by a factor of 2)
xdirs = [] # directories to omit
save = 2
fps = 30
timeout = 120

sift_params = {"sigma": 4, "nOctaveLayers":3, "contrastThreshold":0.03, "edgeThreshold":10}

detector = ShiTomasiDetector(maxCorners=1000)
optical_flow = LucasKanade()
#descriptor = SIFTDescriptor(params=sift_params)
descriptor = ORBDescriptor()
segmenter = FuzzyCMeansSegmenter(5, 0, 10, 2, 2, 10)

############################

boxes = bbox_helper.data2bboxes(data.data)
datas = OrderedDict(sorted(data.data.items()))

try: # this try block ensures the data is saved, even if there's an exception 

    for title, dataset in data.data.items():
        if title in xdirs:
            continue
        
        boxes[title] = OrderedDict(sorted(boxes[title].items()))
        error_tracker_dict = {}

        for class_dir, fbbox in boxes[title].items():

            if class_dir in dataset['xdirs']:
                continue
            
            image = cv2.imread(dataset['url']+class_dir+'/'+"img0000001.jpg", 0)
            
            output_video_path = f"results/video/{time_stamp()}-{class_dir}.mp4"
            frame_height, frame_width = image.shape
            video_writer = T.create_video_writer(output_video_path, (frame_width, frame_height), fps)

            frame = cv2.resize(image, (0,0), fx=1/s, fy=1/s)
            
            gt_csv_path = dataset['url']+'/../annotations/'+class_dir+'.txt'
            error_tracker = T.ErrorTracker(gt_csv_path, scale = s)

            print ("Tracking: ", title, class_dir,)

            all_boxes = bbox_helper.get_all_bboxes(gt_csv_path)

            num_images = len(os.listdir(dataset['url']+class_dir))
                             
            for i in tqdm(range(num_images), desc=f"Processing {class_dir}", unit="frame"):
                try:
                    i += 1
                    file = ("img{0:0"+str(dataset['zc'])+"}.jpg").format(i)

                    image = cv2.imread(dataset['url']+class_dir+'/'+file, 1)
                    frame = cv2.resize(image, (0,0), fx=1/s, fy=1/s)

                    if file == 'img'+(dataset['zc']-1)*'0'+'1.jpg':
                        init_bbox = [int (fbbox[0]/s), int (fbbox[1]/s), int (fbbox[2]/s), int (fbbox[3]/s)]
                        init_bbox = T.list_to_Bbox_obj(init_bbox)
                        tracker = our_tracker.Tracker(first_frame=frame, initial_bbox=init_bbox, feature_detector=detector, optical_flow=optical_flow, feature_descriptor=descriptor, segmenter=segmenter)
                        
                        gt_box = all_boxes[i-1]
                        
                        if save == 1 or save == 2:
                            bbox_obj = T.BoundingBox(int (fbbox[0]/s), int (fbbox[1]/s), int (fbbox[2]/s), int (fbbox[3]/s))
                            error_tracker.update(bbox_obj)
                        continue
                    try:
                        bbox = tracker.track(frame)
                        if math.isnan(all_boxes[i-1][0]) or all_boxes[i-1][0] == 'NaN':
                            gt_box = [-1,-1,0,0]
                        else:
                            gt_box = [x / s for x in all_boxes[i-1]]

                        # Visualization
                        if save == 0 or save == 2:
                            frame = bbox_helper.visualise_bbox(gt_box, T.Bbox_obj_to_list(bbox), file, class_dir, s, dataset['url'])
                            video_writer.write(frame)
                        
                        center_err = error_tracker.update(bbox)
                        runtime = error_tracker.get_running_time()

                        # check if we exceeded runtime, and progress is slow
                        runtime_exceeded = runtime >= timeout and error_tracker.get_recent_fps() < 10
                        
                        if runtime_exceeded: # if computation exceeds timeout, stop and go to next video
                            print("Time limit exceeded, exiting from current video.")
                            break

                    except Exception as e:
                        print("Couldn't find feature points, exiting from current video.")
                        print(e)
                        break

                except ValueError as e: 
                    print("Error: couldn't find feature points, exiting from current video.")
                    print(e)
                    continue

            print(f"Writing video to {output_video_path}.")
            video_writer.release()
            if runtime_exceeded: 
                error_tracker_dict[class_dir] = error_tracker
finally:
    output_file = f"results/Error_info_{time_stamp()}.csv"
    T.write_tracking_info_to_csv(error_tracker_dict, output_file=output_file, error_threshold=20)
    video_writer.release()