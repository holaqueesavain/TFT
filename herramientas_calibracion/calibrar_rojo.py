import cv2
import numpy as np
import mediapipe as mp

def nada(x):
    pass

# Inicializar MediaPipe para la mano
mp_manos = mp.solutions.hands
detector = mp_manos.Hands(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

# Inicializar cámara
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Crear ventana simple con solo 4 controles
cv2.namedWindow('Calibrador Punta Roja', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibrador Punta Roja', 600, 300)

cv2.createTrackbar('Tono Rojo Minimo', 'Calibrador Punta Roja', 0, 20, nada)
cv2.createTrackbar('Saturacion Min (Quitar Piel)', 'Calibrador Punta Roja', 29, 255, nada)
cv2.createTrackbar('Brillo Minimo', 'Calibrador Punta Roja', 137, 255, nada)
cv2.createTrackbar('Circularidad Min', 'Calibrador Punta Roja', 4, 10, nada)
cv2.createTrackbar('Area Minima', 'Calibrador Punta Roja', 15, 200, nada)

print("INSTRUCCIONES SIMPLIFICADAS:")
print("1. El programa detectara TODA tu mano usando los nudillos, no un circulo torpe.")
print("2. 'Saturacion Min': Subela poco a poco (ej. a 80 o 100) para BORRAR TUS DEDOS de la mascara blanca (tu piel es desaturada, la punta es viva).")
print("3. 'Brillo Minimo': Muevelo para borrar sombras que no sean de tu brazo.")
print("4. 'Circularidad Min': Ayuda a ignorar dedos (que son alargados) y quedarse con la punta (redonda).")
print("5. Esc para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    # 1. LEER OPCIONES QUE SE ENTIENDEN
    tono_rojo = cv2.getTrackbarPos('Tono Rojo Minimo', 'Calibrador Punta Roja')
    saturacion_min = cv2.getTrackbarPos('Saturacion Min (Quitar Piel)', 'Calibrador Punta Roja')
    brillo_min = cv2.getTrackbarPos('Brillo Minimo', 'Calibrador Punta Roja')
    circ_min = cv2.getTrackbarPos('Circularidad Min', 'Calibrador Punta Roja') / 10.0
    area_min = cv2.getTrackbarPos('Area Minima', 'Calibrador Punta Roja')

    # Valores fijos que sabemos que funcionan (No marear al usuario con ellos)
    tono_max1 = 10
    tono_min2 = 170

    # 2. DETECTAR EL CONTORNO EXACTO DE LA MANO (Convex Hull)
    res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    puntos_mano = []
    
    if res.multi_hand_landmarks:
        h, w, _ = frame.shape
        # Recorrer todos los 21 nudillos y extraerlos
        for landmark in res.multi_hand_landmarks[0].landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            puntos_mano.append([x, y])

    # 3. FILTRADO DE COLOR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    
    lower1 = np.array([tono_rojo, saturacion_min, brillo_min])
    upper1 = np.array([tono_max1, 255, 255])
    
    lower2 = np.array([tono_min2, saturacion_min, brillo_min])
    upper2 = np.array([180, 255, 255])
    
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))
                          
    # 4. OCULTAR EXACTAMENTE LA MANO (Y MUÑECA/ANTEBRAZO)
    # Si detectó nudillos, hacer la silueta perfecta sobre la mano y extenderla
    if len(puntos_mano) > 0:
        # Calcular el centro aproximado de la mano para saber dirección
        centro_x = sum([p[0] for p in puntos_mano]) / len(puntos_mano)
        centro_y = sum([p[1] for p in puntos_mano]) / len(puntos_mano)
        
        # El punto 0 de MediaPipe suele ser la muñeca
        muneca_x, muneca_y = puntos_mano[0]
        
        # Calcular vector desde el centro hacia la muñeca y alargarlo para tapar brazo
        vec_x = muneca_x - centro_x
        vec_y = muneca_y - centro_y
        
        # Añadir al polígono unos puntos virtuales proyectados hacia atrás (hacia el antebrazo)
        # Factor 1.5 y 2.5 crean un "tubo" hacia el brazo
        pulso_1 = [int(muneca_x + vec_x * 1.5), int(muneca_y + vec_y * 1.5)]
        pulso_2 = [int(muneca_x + vec_x * 2.5), int(muneca_y + vec_y * 2.5)]
        
        # También ensanchamos un poco la muñeca
        puntos_mano.extend([pulso_1, pulso_2])
        # Desplazamientos laterales para engordar el brazo
        puntos_mano.append([pulso_2[0] + 50, pulso_2[1] + 50])
        puntos_mano.append([pulso_2[0] - 50, pulso_2[1] - 50])
        
        puntos_mano = np.array(puntos_mano, dtype=np.int32)
        hull = cv2.convexHull(puntos_mano)
        
        # Rrellenamos de negro el polígono (ahora guantelete largo)
        cv2.drawContours(mask, [hull], 0, 0, -1)           
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_resultado = frame.copy()
    
    # Dibujar la silueta gris en el video normal para ver cómo la envuelve
    if len(puntos_mano) > 0:
        cv2.drawContours(frame_resultado, [hull], 0, (150, 150, 150), 2)
        cv2.putText(frame_resultado, "MANO OCULTADA", tuple(hull[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

    # 5. DETECTAR REDONDO
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Dibujar siempre en azul clarito lo que "vio" como rojo en la máscara para entender fallos
            cv2.drawContours(frame_resultado, [cnt], 0, (255, 255, 0), 1)
            
            # Limite para ignorar ruido pequeño
            if area_min < area < 4000:
                perimetro = cv2.arcLength(cnt, True)
                if perimetro > 0:
                    circularidad = 4 * np.pi * (area / (perimetro * perimetro))
                    
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        tx = int(M["m10"] / M["m00"])
                        ty = int(M["m01"] / M["m00"])
                        
                        # Mostrar el c: 0.XX encima
                        cv2.putText(frame_resultado, f"c:{circularidad:.2f}", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Si pasa nuestro test super exigente: LO HA ENCONTRADO
                        if circularidad > circ_min:
                            cv2.circle(frame_resultado, (tx, ty), 6, (0, 255, 0), -1)

    cv2.imshow('Mascara Color (Negro = tu mano)', mask)
    cv2.imshow('Calibrador Punta Roja', frame_resultado)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
