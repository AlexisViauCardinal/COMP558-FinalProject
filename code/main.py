import utils.bbox_helper as bbox_helper
import data.datasets as data
from collections import OrderedDict
import numpy as np
import os
import cv2
import time
import math
import utils.test_utils as T
import os
import tracker as our_tracker

from feature_detection.shi_tomasi import ShiTomasiDetector
from optical_flow.lucas_kanade import LucasKanade
from feature_description.sift_descriptor import SIFTDescriptor
from segmentation.fuzzy_c_means import FuzzyCMeansSegmenter

############ Params ############
s = 1
xdirs = []
save = 2

sift_params = {"sigma": 4, "nOctaveLayers":3, "contrastThreshold":0.03, "edgeThreshold":10}
# sift_params = {}

detector = ShiTomasiDetector()
optical_flow = LucasKanade()
descriptor = SIFTDescriptor(params=sift_params)
segmenter = FuzzyCMeansSegmenter(5, 0, 10, 2, 2, 10)


boxes = bbox_helper.data2bboxes(data.data)
datas = OrderedDict(sorted(data.data.items()))

for title, dataset in data.data.items():
    if title in xdirs:
        continue
    
    boxes[title] = OrderedDict(sorted(boxes[title].items()))
    error_tracker_dict = {}
    for class_dir, fbbox in boxes[title].items():
        if class_dir in dataset['xdirs']:
            continue
        
        file = ("img0000001.jpg")
        image = cv2.imread(dataset['url']+class_dir+'/'+file, 0)
        frame = cv2.resize(image, (0,0), fx=1/s, fy=1/s)
        
        output_video_path = f"video/{class_dir}.mp4"
        frame_height, frame_width = frame.shape
        fps = 30
        video_writer = T.create_video_writer(output_video_path, (frame_width, frame_height), fps)

        gt_csv_path = dataset['url']+'/../annotations/'+class_dir+'.txt'
        error_tracker = T.ErrorTracker(gt_csv_path)

        print (title, class_dir)

        all_boxes = bbox_helper.get_all_bboxes(gt_csv_path)
        print("Num images in folder: ",len(os.listdir(dataset['url']+class_dir)))
        for i in np.arange(len(os.listdir(dataset['url']+class_dir))):
            i += 1
            #file = ("{0:0"+str(dataset['zc'])+"}.jpg").format(i)
            file = ("img{0:0"+str(dataset['zc'])+"}.jpg").format(i)

            image = cv2.imread(dataset['url']+class_dir+'/'+file, 1)
            frame = cv2.resize(image, (0,0), fx=1/s, fy=1/s)
            # D.showImage(frame, 'frame')
            #if file == (dataset['zc']-1)*'0'+'1.jpg':
            if file == 'img'+(dataset['zc']-1)*'0'+'1.jpg':
                init_bbox = [int (fbbox[0]/s), int (fbbox[1]/s), int (fbbox[2]/s), int (fbbox[3]/s)]
                init_bbox = T.list_to_Bbox_obj(init_bbox)
                tracker = our_tracker.Tracker(first_frame=frame, initial_bbox=init_bbox, feature_detector=detector, optical_flow=optical_flow, feature_descriptor=descriptor, segmenter=segmenter)
                
                gt_box = all_boxes[i-1]
                
                if save == 1:
                    bbox_obj = T.BoundingBox(int (fbbox[0]/s), int (fbbox[1]/s), int (fbbox[2]/s), int (fbbox[3]/s))
                    error_tracker.update(bbox_obj)
                continue
            try:
                bbox = tracker.track(frame)
                if math.isnan(all_boxes[i-1][0]) or all_boxes[i-1][0] == 'NaN':
                    gt_box = [-1,-1,0,0]
                else:
                    gt_box = all_boxes[i-1]

                # Visulization
                if save == 0 or save == 2:
                    frame = bbox_helper.visualise_bbox(gt_box, T.Bbox_obj_to_list(bbox), file, class_dir, s, dataset['url'])
                    # Display the frame with bounding boxes
                    #cv2.imshow(class_dir, frame)
                    video_writer.write(frame)
                    #k = cv2.waitKey(5) & 0xFF 
                    #if k == 27:
                    #    break

                if save == 1  or save == 2:
                    error_tracker.update(bbox)
            except: 
                print("Lost tracking.")

        print("Releasing Writer, writing to dict.")
        video_writer.release()
        error_tracker_dict[class_dir] = error_tracker
    output_file = "Error_info.csv"
    T.write_tracking_info_to_csv(error_tracker_dict, output_file=output_file, error_threshold=20)