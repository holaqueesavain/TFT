import cv2

print("Iniciando escaneo de puertos USB...")
for i in range(10):  # Probamos del 0 al 9
    # Quitamos el CAP_DSHOW un momento para ver si es eso lo que bloquea
    cap = cv2.VideoCapture(i) 
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ ¡ÉXITO! Cámara detectada y leyendo vídeo en el ID: {i}")
        else:
            print(f"⚠️ Cámara detectada en ID {i}, pero no devuelve imagen (quizás saturada).")
        cap.release()
print("Escaneo terminado.")