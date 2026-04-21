import mediapipe as mp
import cv2
try:
    print("Contenido de mp:", dir(mp))
    print("¿Existe solutions?", hasattr(mp, 'solutions'))
    if hasattr(mp, 'solutions'):
        print("¡ÉXITO! Solutions encontrado.")
    else:
        print("FALLO: Solutions sigue ausente.")
except Exception as e:
    print(f"Error: {e}")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara {i} detectada")
        cap.release()
    else:
        print(f"Cámara {i} NO detectada")
for i in range(5):  # prueba índices del 0 al 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara detectada en índice {i}")
        cap.release()
    else:
        print(f"Cámara NO detectada en índice {i}")