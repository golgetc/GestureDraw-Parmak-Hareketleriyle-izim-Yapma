import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Çizim için renkleri ve silgiyi belirle
colors = {'blue': (255, 0, 0), 'red': (0, 0, 255), 'green': (0, 255, 0), 'eraser': (0, 0, 0)}
color = colors['blue']

# Çizim yapılacak resmi başlat
image = np.zeros((480, 640, 3), np.uint8)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # MediaPipe Hands modeli RGB görüntüleri işler
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        # El tespiti ve takip
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Parmak ucu koordinatını al
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                
                # Çizim yap
                cv2.circle(image, (x, y), 5, color, -1)

        # Görüntüyü yeniden BGR'ye çevir
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Ekranı göster
        cv2.imshow('El Takibi', frame)
        cv2.imshow('Elin Çizimi', image)

        # Tuşlara basıldığında renk seçimi
        key = cv2.waitKey(5)
        if key == ord('r'):
            color = colors['red']
        elif key == ord('b'):
            color = colors['blue']
        elif key == ord('g'):
            color = colors['green']
        elif key == ord('e'):
            color = colors['eraser']
        elif key == ord('c'):  # Ekranı temizle
            image = np.zeros((480, 640, 3), np.uint8)
        elif key == ord('q'):  # Programı kapat
            break

cap.release()
cv2.destroyAllWindows()
