"""
Prueba de detección de manos con YOLOv8.
"""

import cv2
import time
import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(script_dir, 'yolov8n-hands.pt')
modelo = YOLO(modelo_path)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

t_prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    results = modelo.predict(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, "Mano (YOLO)", (x1, y1 - 10), 0, 0.6, (0, 0, 255), 2)

    t_now = time.time()
    fps = 1 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0
    t_prev = t_now
    cv2.putText(frame, f"Rendimiento: {int(fps)} FPS", (10, frame.shape[0] - 20), 0, 0.6, (0, 255, 0), 2)

    cv2.imshow('Test Comparativo: YOLO v8', frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()