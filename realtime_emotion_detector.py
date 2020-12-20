import cv2
import onnxruntime as ort
import argparse, copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#This is for mac you can comment it if you want
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow.keras as keras
from  tensorflow.keras.layers import *

model = keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])



model.load_weights('/Users/Kunal/Downloads/emoji_weights_v1.h5')



face_detector_onnx = "/Users/Kunal/Downloads/version-RFB-320.onnx"
face_detector      = ort.InferenceSession(face_detector_onnx)


def predict_class(roi_img, model):
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_img, (48,48)), -1),0)
    pred_class = model.predict_classes(cropped_img)
    return emotion_dict[pred_class[0]]

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
