import os
import cv2
import time
import mxnet as mx
from . import predict
from .config_farm import configuration_10_320_20L_5scales_v2 as cfg
from threading import Thread

model_file_path = "LFFD/models/train_10_320_20L_5scales_v2_iter_1000000.params" 
symbol_file_path = "LFFD/symbols/symbol_10_320_20L_5scales_v2_deploy.json"
ctx = mx.gpu(0)

### Threshold for Face Matching
cosine_threshold = 0.8
proba_threshold = 0.6
comparing_num = 5
face_predictor = predict.Predict(mxnet=mx,
                    symbol_file_path=symbol_file_path,
                    model_file_path=model_file_path,
                    ctx=ctx,
                    receptive_field_list=cfg.param_receptive_field_list,
                    receptive_field_stride=cfg.param_receptive_field_stride,
                    bbox_small_list=cfg.param_bbox_small_list,
                    bbox_large_list=cfg.param_bbox_large_list,
                    receptive_field_center_start=cfg.param_receptive_field_center_start,
                    num_output_scales=cfg.param_num_output_scales)

class DetectFaces():
    def __init__(self, frame = None):
        ### Loading Face Detection Models
        self.frame = frame
        self.stopped = False
       

    def start(self):
        print("START VIDEO PROCESSSING")
        Thread(target=self.drawBBoxes, args=()).start()
        Thread.daemon = True
        return self

    def drawBBoxes(self):
        while not self.stopped:
            frame = self.frame
            bboxes, infer_time = face_predictor.predict(frame, resize_scale=1, score_threshold=0.6, top_k=10000, \
                                                           NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[]) 
            
            h, w, c = frame.shape
            for bbox in bboxes:
                print(bbox)
                bbox_int = [int(b) for b in bbox[:-1]]
                bbox_int[0] = max(0, min(w - 1, bbox_int[0]))
                bbox_int[1] = max(0, min(h - 1, bbox_int[1]))
                bbox_int[2] = max(0, min(w - 1, bbox_int[2]))
                bbox_int[3] = max(0, min(h - 1, bbox_int[3]))
                cv2.rectangle(frame, tuple(bbox_int[0:2]), tuple(bbox_int[2:4]), (0, 255, 0), 2)
                self.frame = frame
    
    def stop(self):
        self.stopped = True