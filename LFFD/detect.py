import time
import mxnet as mx
from . import predict

class DetectFaces():
    def __init__(self, frame = None):
        ### Loading Face Detection Models
        self.frame = frame
        model_file_path = "models/train_10_320_20L_5scales_v2_iter_1000000.params" 
        symbol_file_path = "symbols/symbol_10_320_20L_5scales_v2_deploy.json"
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

    def drawBBoxes(self):
        print("Getting Bounding Boxes")
