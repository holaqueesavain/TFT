import cv2
import numpy as np
import glob
import os

# --- Configuración ---
chessboard_cols = 7   # número de esquinas internas en ancho
chessboard_rows = 7   # número de esquinas internas en alto
square_size = 0.04    # tamaño del cuadrado en METROS (4 cm = 0.04 m)

# --- Preparar puntos del mundo real ---
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2) * square_size

# Listas para guardar puntos
objpoints = []       # puntos 3D reales
imgpoints_l = []     # puntos 2D izquierda
imgpoints_r = []     # puntos 2D derecha

# Cargar imágenes
images_left = sorted(glob.glob('calibracion_estereo/calib_images/left*.jpg'))
images_right = sorted(glob.glob('calibracion_estereo/calib_images/right*.jpg'))

if len(images_left) == 0 or len(images_right) == 0:
    print("No se encontraron imágenes. Revisa la carpeta 'calib_images'.")
    exit()

print(f"Encontradas {len(images_left)} imágenes izquierdas y {len(images_right)} derechas.")

for i in range(len(images_left)):
    img_l = cv2.imread(images_left[i])
    img_r = cv2.imread(images_right[i])
    
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Buscar esquinas del tablero
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (chessboard_cols, chessboard_rows), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (chessboard_cols, chessboard_rows), None)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        print(f"Par {i+1} detectado")
    else:
        print(f"Par {i+1} NO válido")

if len(objpoints) < 10:
    print("Necesitas al menos 10 pares buenos. Vuelve a tomar fotos.")
    exit()

# Calibración estéreo
print("Calibrando...")
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    None, None, None, None,
    gray_l.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Rectificación
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, gray_l.shape[::-1], R, T)

# Guardar resultados
np.savez('calibracion_estereo/calibracion_estereo.npz', K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("¡Calibración terminada!")
print("Archivo calibracion_estereo.npz' guardado.")
print("Ahora puedes usarlo en tu código de profundidad 3D.")