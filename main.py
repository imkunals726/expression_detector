import cv2
import numpy as np
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from  tensorflow.keras.layers import *

def predict_class(roi_img, model):
    #print(roi_img,'roi_img')i
    
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_img, (48,48)), -1),0)
    cropped_img = cropped_img / 255.0
    pred_class = model.predict_classes(cropped_img)
    return emotion_dict[pred_class[0]]

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



model.load_weights('emoji_weights_v1.h5')


detector = MTCNN()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_faces(img)
    faces = [detector.detect_faces(img)[0]['box']] if faces else []

    for (x, y , w ,h) in faces:
        if w > 50:
            roi_img    = gray[y:y + h, x:x + w]
            class_name = predict_class(roi_img, model)
            
            cv2.putText(img, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
        break;

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
