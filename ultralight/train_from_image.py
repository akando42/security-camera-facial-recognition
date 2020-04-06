import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
from utils import area_of, iou_of , hard_nms, predict 

onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

TRAINING_BASE = "data/minh/camera"

dirs = os.listdir(TRAINING_BASE)
images = []
names = []

### Reading Images from Training Folder
for label in dirs:
    for i, raw_img in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))):
        print("{}_image_{}".format(label, raw_img))
        gray = cv2.cv2Color(raw_img, cv2.COLOR_BGR2GRAY)
        aligned_face = fa.align(raw_img, gray, d)


### Generating Embedding from Images
with tf.Graph().as_default():
    with tf.Session() as sess:
        print("loading checkpoint ...")
        saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
        saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
        embeds = sess.run(embeddings, feed_dict=feed_dict)
        with open("embeddings/embeddings.pkl", "wb") as f:
            pickle.dump((embeds, names), f)
        print("Done!")
