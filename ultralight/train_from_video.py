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


TRAINING_BASE = '../data/test'

dirs = os.listdir(TRAINING_BASE)
images = []
names = []

for label in dirs:
    for i, fn in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))):
        print(f"start collecting faces from {label}'s data")
        cap = cv2.VideoCapture(os.path.join(TRAINING_BASE, label, fn))
        frame_count = 0
        while True:
            # read video frame
            ret, raw_img = cap.read()
            # process every 5 frames
            if frame_count % 5 == 0 and raw_img is not None:
                h, w, _ = raw_img.shape
                img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 480))
                img_mean = np.array([127, 127, 127])
                img = (img - img_mean) / 128
                img = np.transpose(img, [2, 0, 1])
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32)

                confidences, boxes = ort_session.run(None, {input_name: img})
                boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                # if face detected
                if boxes.shape[0] > 0:
                    x1, y1, x2, y2 = boxes[0,:]
                    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                    aligned_face = fa.align(raw_img, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                    aligned_face = cv2.resize(aligned_face, (112,112))

                    cv2.imwrite(f'faces/tmp/{label}_{frame_count}.jpg', aligned_face)

                    aligned_face = aligned_face - 127.5
                    aligned_face = aligned_face * 0.0078125
                    images.append(aligned_face)
                    names.append(label)

            frame_count += 1
            if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break

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
