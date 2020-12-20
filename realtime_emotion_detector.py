import cv2
import onnxruntime            as ort
import argparse, copy
import numpy                  as np
import matplotlib.pyplot      as plt
import tensorflow             as tf
import os
import tensorflow.keras       as keras
from   tensorflow.keras.layers import *
from   emoji_model             import get_emoji_model
from   utils 		       import predict_class
#This is for mac you can comment it if you want
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model              = get_emoji_model(pretrained=True)
face_detector_onnx = "version-RFB-320.onnx"
face_detector      = ort.InferenceSession(face_detector_onnx)

def face_box(image):
    height   = image.shape[0]
    width    = image.shape[1]
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))

    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})


    new_boxes = boxes[0]
    indices = tf.image.non_max_suppression(new_boxes, confidences[0][:,1], 1, 0.5 )

    selected_boxes = tf.gather(new_boxes, indices)
    selected_boxes = selected_boxes.numpy()

    selected_boxes[:,0] *= width
    selected_boxes[:,1] *= height
    selected_boxes[:,2] *= width
    selected_boxes[:,3] *= height


    x_min, y_min, x_max, y_max = selected_boxes[0]
    return x_min, y_min, x_max, y_max

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, img = cap.read()

    x_min, y_min, x_max, y_max = face_box(img)
    if min([x_min, y_min, x_max, y_max] ) >0:
        cv2.rectangle(img, (x_min,y_min), (x_max, x_max), (255, 0 , 0), 3)
        gray            = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                    
        roi_img         = gray[int(y_min):int(y_max), int(x_min):int(x_max)]
        #print(roi_img.shape)
        predicted_class = predict_class(roi_img, model)
        cv2.putText(img, predicted_class, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
