import cv2
import numpy as np

def sincronizar_barra(x):
    pass

# Inicializar cámaras
cam_izq = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam_der = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Crear ventana de control
cv2.namedWindow('Controles')
cv2.resizeWindow('Controles', 600, 300)
cv2.createTrackbar('Verde Min', 'Controles', 38, 179, sincronizar_barra)
cv2.createTrackbar('Verde Max', 'Controles', 85, 179, sincronizar_barra)
cv2.createTrackbar('Sat Min', 'Controles', 18, 255, sincronizar_barra)
cv2.createTrackbar('Val Min', 'Controles', 107, 255, sincronizar_barra)
cv2.createTrackbar('Circ Min', 'Controles', 2, 10, sincronizar_barra)
cv2.createTrackbar('Area Min', 'Controles', 10, 500, sincronizar_barra)

def procesar_vision(frame, hmin, hmax, smin, vmin, cmin, amin):
    """
    Ajustar los parámetros hasta detectar el círculo verde del robot en ambas cámaras.
    Mantiene concordancia con la lógica de seguimiento.py (CLAHE + Morph + Circularidad).
    """
    # 1. Filtro de Color (HSV + CLAHE para consistencia con el entorno real)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    mask = cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, 255, 255]))
    
    # 2. Limpieza de ruido (Apertura y Cierre)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    # 3. Detección del objeto circular mas grande
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = frame.copy()
    centro = None
    
    if conts:
        # Ordenar por area y filtrar por circularidad
        for cnt in sorted(conts, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < amin: continue
            
            perim = cv2.arcLength(cnt, True)
            if perim > 0:
                circ = (4 * np.pi * area) / (perim ** 2)
            else:
                circ = 0
            
            # Dibujamos en azul cian todos los candidatos potenciales
            cv2.drawContours(res, [cnt], 0, (255, 255, 0), 1)
            if circ > cmin:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centro = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                    # Dibujamos un círculo sólido verde en el objeto ganador
                    cv2.circle(res, centro, 6, (0, 255, 0), -1)
                    break 
                    
    return res, mask

print("Instrucciones: Mueve las barras hasta que el robot sea detectado. Pulsa ESC para salir.")

while True:
    ri, fi = cam_izq.read()
    rd, fd = cam_der.read()
    if not ri or not rd: break
    
    fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)
    
    # Obtener valores actuales de las barras
    h_m = cv2.getTrackbarPos('Verde Min', 'Controles')
    h_M = cv2.getTrackbarPos('Verde Max', 'Controles')
    s_m = cv2.getTrackbarPos('Sat Min', 'Controles')
    v_m = cv2.getTrackbarPos('Val Min', 'Controles')
    c_m = cv2.getTrackbarPos('Circ Min', 'Controles') / 10.0
    a_m = cv2.getTrackbarPos('Area Min', 'Controles')
    
    resi, maski = procesar_vision(fi, h_m, h_M, s_m, v_m, c_m, a_m)
    resd, maskd = procesar_vision(fd, h_m, h_M, s_m, v_m, c_m, a_m)
    
    # Pegar Imagen Original | Mascara de color
    izq = np.hstack((resi, cv2.cvtColor(maski, cv2.COLOR_GRAY2BGR)))
    der = np.hstack((resd, cv2.cvtColor(maskd, cv2.COLOR_GRAY2BGR)))
    
    cv2.imshow('Calibracion IZQUIERDA', cv2.resize(izq, (800, 300)))
    cv2.imshow('Calibracion DERECHA', cv2.resize(der, (800, 300)))
    
    if cv2.waitKey(1) == 27: break

cam_izq.release()
cam_der.release()
cv2.destroyAllWindows()
