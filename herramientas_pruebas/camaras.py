"""
Prueba para comprobar qué cámaras están activas y funcionando.
"""

import cv2


for i in range(10):
    cap = cv2.VideoCapture(i) 
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"ID {i}: Cámara disponible y funcionando.")
        else:
            print(f"ID {i}: Dispositivo detectado pero sin flujo de imagen.")
        cap.release()

