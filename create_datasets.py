import os

import pickle
import mediapipe as mp
import cv2

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing.utils
mp_drawing_styles = mp.solutions.drawing.styles

hand_detector_model = mp.hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data_directory = 'D:\Machine Learning Projects\Sign-Language-Detection'

dataset_list = []
labels_list = []

for directories_ in os.listdir(data_directory):
    for image_path in os.listdir(os.path.join(data_directory, directories_)):
        data_aux = []

        store_x = []
        store_y = []

        image = cv2.imread(os.path.join(data_directory, directories_))
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hand_detector_model.process(image_RGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for count in range(len(hand_landmarks.landmarks)):
                    x = hand_landmarks.landmarks[count].x
                    y = hand_landmarks.landmarks[count].y

                    store_x.append(x)
                    store_y.append(y)

                for count in range(len(hand_landmarks.landmarks)):
                    x = hand_landmarks.landmarks[count].x
                    y = hand_landmarks.landmarks[count].y

                    data_aux.append(x - min(store_x))
                    data_aux.append(y - store_y)
                
            dataset_list.append(data_aux)
            labels_list.append(directories_)

data_file = open('dataset_list.pickle', 'wb')
pickle.dump({'dataset_list"' : dataset_list, 'labels_list' : labels_list})
data_file.close()

