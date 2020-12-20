import numpy as np, cv2

def predict_class(roi_img, model):
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_img, (48,48)), -1),0)
    pred_class = model.predict_classes(cropped_img)
    return emotion_dict[pred_class[0]]
