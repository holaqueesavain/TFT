"""
Prueba de segmentación por color en HSV para ver el ruido que hay en el entorno del laboratorio.
Se compara la detección del color verde del robot con el rojo de objetos de prueba.
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

lower_verde = np.array([38, 18, 107])
upper_verde = np.array([85, 255, 255])

lower_rojo1 = np.array([0, 70, 50])
upper_rojo1 = np.array([10, 255, 255])
lower_rojo2 = np.array([170, 70, 50])
upper_rojo2 = np.array([180, 255, 255])

print("Prueba de Color activa - Teclas: 'v' Verde, 'r' Rojo, 'ESC' Salir")

modo = 'verde' 
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if modo == 'verde':
        mask = cv2.inRange(hsv, lower_verde, upper_verde)
        color = (0, 255, 0)
        label = "MARCADOR ROBOT (VERDE)"
    else:
        m1 = cv2.inRange(hsv, lower_rojo1, upper_rojo1)
        m2 = cv2.inRange(hsv, lower_rojo2, upper_rojo2)
        mask = cv2.bitwise_or(m1, m2)
        color = (0, 0, 255)
        label = "OBJETO PRUEBA (ROJO)"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ruido = len([c for c in contours if cv2.contourArea(c) < 500])

    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, f"Modo: {label}", (20, 40), 0, 0.7, color, 2)
    cv2.putText(display, f"Objetos descartados (ruido): {ruido}", (20, 80), 0, 0.6, color, 1)

    cv2.imshow("Prueba de Segmentacion por Color", display)
    cv2.imshow("Imagen Original", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('v'): modo = 'verde'
    elif key == ord('r'): modo = 'rojo'

cap.release()
cv2.destroyAllWindows()
