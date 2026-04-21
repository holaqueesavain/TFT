import cv2
import os

cap = cv2.VideoCapture(2) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

output_folder = "capturas"
os.makedirs(output_folder, exist_ok=True)

counter = 1
print("Presiona 'c' para capturar, 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo cámara")
        break

    cv2.imshow("Camara (640x480)", frame)
    key = cv2.waitKey(1) & 0xFF

    # salir
    if key == ord('q'):
        break

    # capturar
    if key == ord('c'):
        filename = f"Img{counter:d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"Capturada: {filepath}")
        counter += 1

cap.release()
cv2.destroyAllWindows()