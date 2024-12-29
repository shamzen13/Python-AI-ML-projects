import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./signLang/model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)

            for i in range(len(hand_landmarks.landmark)):
                x = (hand_landmarks.landmark[i].x - x_min) / (x_max - x_min)
                y = (hand_landmarks.landmark[i].y - y_min) / (y_max - y_min)
                data_aux.append(x)
                data_aux.append(y)

            x1 = int(x_min * W) - 10
            y1 = int(y_min * H) - 10
            x2 = int(x_max * W) + 10
            y2 = int(y_max * H) + 10

            probabilities = model.predict_proba([np.asarray(data_aux)])
            print(f"Prediction probabilities: {probabilities}")
            if max(probabilities[0]) > 0.5:  # Confidence threshold
                predicted_character = labels_dict[int(np.argmax(probabilities))]
            else:
                predicted_character = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
