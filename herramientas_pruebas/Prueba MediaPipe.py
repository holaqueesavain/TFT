import cv2
import mediapipe as mp


mp_manos = mp.solutions.hands
detector = mp_manos.Hands(min_detection_confidence=0.7, max_num_hands=1)
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camara.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break
    
    frame = cv2.flip(frame, 1)
    
    res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        landmarks = res.multi_hand_landmarks[0].landmark
        h, w, _ = frame.shape

        # Extraemos las coordenadas de la Muñeca (0) y base de los dedos (5 y 17)
        x0, y0 = int(landmarks[0].x * w), int(landmarks[0].y * h)
        x5, y5 = int(landmarks[5].x * w), int(landmarks[5].y * h)
        x17, y17 = int(landmarks[17].x * w), int(landmarks[17].y * h)

       
        cx, cy = int((x0 + x5 + x17) / 3), int((y0 + y5 + y17) / 3)

        cv2.line(frame, (x0, y0), (x5, y5), (255, 0, 0), 2)
        cv2.line(frame, (x5, y5), (x17, y17), (255, 0, 0), 2)
        cv2.line(frame, (x17, y17), (x0, y0), (255, 0, 0), 2)

        cv2.circle(frame, (x0, y0), 6, (0, 0, 255), -1)
        cv2.circle(frame, (x5, y5), 6, (0, 0, 255), -1)
        cv2.circle(frame, (x17, y17), 6, (0, 0, 255), -1)

        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

        cv2.putText(frame, "Punto 0", (x0+10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Punto 5", (x5+10, y5-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Punto 17", (x17+10, y17-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Centroide", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("Captura MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camara.release()
cv2.destroyAllWindows()