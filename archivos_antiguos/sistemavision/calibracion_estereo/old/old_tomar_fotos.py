import cv2
import os

# Crear carpeta si no existe
os.makedirs('calibracion_estereo/calib_images', exist_ok=True)

# Inicializar cámaras (ajustar índices según sea necesario)
capL = cv2.VideoCapture(1)  # izquierda
capR = cv2.VideoCapture(2)  # derecha
capR.set(cv2.CAP_PROP_FOCUS, 0) 
capL.set(cv2.CAP_PROP_FOCUS, 0) #foco lo mas lejano posible
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640   )
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 )
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640   )
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 )

# Mostrar ventanas
cv2.namedWindow('Izquierda', cv2.WINDOW_NORMAL)
cv2.namedWindow('Derecha', cv2.WINDOW_NORMAL)

count = 0
print("PAR FOTO Presiona 'c' para capturar un par de fotos")
print("SALIR Presiona 'q' para salir")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Error al leer cámaras. Revisa conexiones.")
        break

    # Mostrar imágenes
    cv2.imshow('Izquierda', frameL)
    cv2.imshow('Derecha', frameR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Guardar fotos
        filenameL = f'calibracion_estereo/calib_images/left_{count:02d}.jpg'
        filenameR = f'calibracion_estereo/calib_images/right_{count:02d}.jpg'
        cv2.imwrite(filenameL, frameL)
        cv2.imwrite(filenameR, frameR)
        print(f"Foto {count} guardada: {filenameL}, {filenameR}")
        count += 1
    elif key == ord('q'):
        break

# Liberar recursos
capL.release()
capR.release()
cv2.destroyAllWindows()