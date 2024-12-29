#creating dataset

import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle

DATA_DIR = './signLang/data'

#objects to help landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#object to detect hand
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3 )

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):

        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        #convert image to rgb to input it into mediapipe (landmark detection is in rgb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        #create an array of landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x  = hand_landmarks.landmark[i].x
                    y  = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(dir_)


f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels},f)
f.close()

