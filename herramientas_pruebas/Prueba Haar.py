import cv2
import os
import sys

# Carga de clasificador Haar
script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(script_dir, 'mano_haar.xml')

if not os.path.exists(ruta_modelo):
    print(f"Error: No se encuentra {ruta_modelo}")
    sys.exit(1)

cascade = cv2.CascadeClassifier(ruta_modelo)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("Iniciando prueba HAAR - Pulsa ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # Procesamiento en gris para Haar
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecciones = cascade.detectMultiScale(gris, 1.1, 5, minSize=(50, 50))
    
    for (x, y, w, h) in detecciones:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, 'Deteccion Haar', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, "PRUEBA HAAR: Deteccion por cascada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Prueba Haar Cascade", frame)
    
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
