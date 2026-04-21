import cv2
import mediapipe as mp
import numpy as np


# MediaPipe manos
mp_hands = mp.solutions.hands
detector_manos = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3,  
    max_num_hands=2
)

# Parámetros estereoscópicos
BASELINE = 0.093      
FOCAL = 849.0
CX = 320.0
CY = 240.0

# Matriz Q 
Q = np.float32([
    [1, 0, 0, -CX],
    [0, 1, 0, -CY],
    [0, 0, 0, FOCAL],
    [0, 0, -1.0/BASELINE, 0]
])

# Filtro suavizado
ALPHA = 0.5

# Seguridad y Límites de trabajo 
LIM_X_MIN, LIM_X_MAX = 0.0, 0.0
LIM_Z_MIN, LIM_Z_MAX = 0.0, 0.0
OFFSET_Y = 0.30

# Motor de Visión Estéreo SGBM (Para el mapa de profundidad)
stereo = cv2.StereoSGBM_create(
    minDisparity=17,
    numDisparities=112,
    blockSize=7,
    P1=1000,
    P2=4000,
    disp12MaxDiff=1,
    uniquenessRatio=0,
    speckleWindowSize=60,
    speckleRange=12
)


puntos_actuales = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_actuales.append((x, y))

def capturar_puntos(camera, window_name, num_puntos=4):
    """Captura tornillos con clics del ratón en la ventana de la cámara."""
    puntos_actuales.clear() 
    puntos = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    while len(puntos) < num_puntos:
        ret, frame = camera.read()
        if not ret: break

        # Guardamos los clics nuevos
        while len(puntos) < len(puntos_actuales):
            puntos.append(puntos_actuales[len(puntos)])

        # Dibujar los puntos que ya se han clicado
        for p in puntos:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == 27: break  

    cv2.destroyWindow(window_name)
    return puntos

def mano_cerrada(landmarks):
    """Detecta si la mano está cerrada analizando si las puntas de los dedos están caídas."""
    puntas = [8, 12, 16, 20]
    nudillos = [6, 10, 14, 18]
    dedos_doblados = 0
    
    px_base = landmarks[0].x
    py_base = landmarks[0].y
    
    for i in range(4):
        dist_punta = (landmarks[puntas[i]].x - px_base)**2 + (landmarks[puntas[i]].y - py_base)**2
        dist_nudillo = (landmarks[nudillos[i]].x - px_base)**2 + (landmarks[nudillos[i]].y - py_base)**2
        
        # Si la punta está más cerca de la muñeca que el nudillo medio, el dedo está doblado haca dentro
        if dist_punta < dist_nudillo:
            dedos_doblados += 1
            
    return dedos_doblados

def obtener_centro_mano(landmarks):
    """Devuelve el centro aproximado de la mano promediando 3 puntos clave."""
    x = int((landmarks[0].x*640 +
             landmarks[5].x*640 +
             landmarks[17].x*640) / 3)

    y = int((landmarks[0].y*480 +
             landmarks[5].y*480 +
             landmarks[17].y*480) / 3)

    return np.clip(x, 0, 639), np.clip(y, 0, 479)


def detectar_punta_verde(frame, hand_landmarks=None):
    """Detecta la punta verde del robot aplicando filtros mágicos de seguridad."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    lower = np.array([38, 18, 107])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Ocultar la mano, una vez detectada
    if hand_landmarks is not None:
        h, w, _ = frame.shape
        puntos = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            puntos.append([x, y])
        if len(puntos) > 0:
            hull = cv2.convexHull(np.array(puntos, dtype = np.int32))
            cv2.drawContours(mask, [hull], 0, 0, -1)  
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Buscar Contornos y filtrar
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 10:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return np.clip(cx, 0, 639), np.clip(cy, 0, 479)
                
    return None


def triangular(p_izq, p_der, tolerancia_y, estado_anterior=None):
    """
    Convierte dos puntos 2D en coordenadas 3D reales.
    Si estado_anterior es None, no aplica suavizado (ideal para calibrar).
    """
    if p_izq and p_der:
        if abs(p_izq[1] - p_der[1]) < tolerancia_y:
            disp = (640 - p_izq[0]) - (640 - p_der[0])
            if abs(disp) > 1:
                vec = np.dot(Q, np.array([640 - p_izq[0], p_izq[1], disp, 1.0]))
                if vec[3] != 0:
                    X = vec[0] / vec[3]
                    Y = vec[1] / vec[3]
                    Z = abs(vec[2] / vec[3])

                    
                    if estado_anterior is not None and estado_anterior != (0,0,0):
                        x_p, y_p, z_p = estado_anterior
                        X = ALPHA * X + (1 - ALPHA) * x_p
                        Y = ALPHA * Y + (1 - ALPHA) * y_p
                        Z = ALPHA * Z + (1 - ALPHA) * z_p

                    return (X, Y, Z), (X, Y, Z)

    return None, estado_anterior



if __name__ == "__main__":
    cam_izq = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam_der = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Captura los 4 tornillos
    p_izq = capturar_puntos(cam_izq, "Calibracion Izquierda")
    p_der = capturar_puntos(cam_der, "Calibracion Derecha")

    # Triangula los puntos 3D y extrae X y Z
    if len(p_izq) == 4 and len(p_der) == 4:
        coords_3d = []
        for i in range(4):
            punto_3d, _ = triangular(p_izq[i], p_der[i], 500, None)
            if punto_3d is not None:
                coords_3d.append(punto_3d)

        if len(coords_3d) >= 2:
            lista_x = [p[0] for p in coords_3d]
            lista_z = [p[2] for p in coords_3d]
            LIM_X_MIN, LIM_X_MAX = min(lista_x), max(lista_x)
            LIM_Z_MIN, LIM_Z_MAX = min(lista_z), max(lista_z)
        else:
            LIM_X_MIN, LIM_X_MAX, LIM_Z_MIN, LIM_Z_MAX = -0.25, 0.25, 0.30, 0.80
    else:
        LIM_X_MIN, LIM_X_MAX, LIM_Z_MIN, LIM_Z_MAX = -0.25, 0.25, 0.30, 0.80

    estado_mano = (0, 0, 0)
    estado_robot = (0, 0, 0)

    print("--- SISTEMA INICIADO (MODO MANUAL V1) ---")

    while True:
        ret_i, frame_i = cam_izq.read()
        ret_d, frame_d = cam_der.read()
        if not ret_i or not ret_d: break

        frame_i = cv2.flip(frame_i, 1)
        frame_d = cv2.flip(frame_d, 1)

        res_i = detector_manos.process(cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB))
        res_d = detector_manos.process(cv2.cvtColor(frame_d, cv2.COLOR_BGR2RGB))

        punto_mano_i = obtener_centro_mano(res_i.multi_hand_landmarks[0].landmark) if res_i.multi_hand_landmarks else None
        punto_mano_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark) if res_d.multi_hand_landmarks and len(res_d.multi_hand_landmarks)==1 else None

        punto_robot_i = detectar_punta_verde(frame_i, hand_landmarks=res_i.multi_hand_landmarks[0] if res_i.multi_hand_landmarks else None)
        punto_robot_d = detectar_punta_verde(frame_d)

        if punto_mano_i: cv2.circle(frame_i, punto_mano_i, 5, (255,0,0), -1) 
        if punto_robot_i: cv2.circle(frame_i, punto_robot_i, 6, (0,255,0), -1) 

        mano_xyz, estado_mano = triangular(punto_mano_i, punto_mano_d, 200, estado_mano)
        robot_xyz, estado_robot = triangular(punto_robot_i, punto_robot_d, 300, estado_robot)

        if mano_xyz:
            xm, ym, zm = mano_xyz
            if xm < LIM_X_MIN or xm > LIM_X_MAX or zm < LIM_Z_MIN or zm > LIM_Z_MAX:
                cv2.putText(frame_i, "LIMITE ALCANZADO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            cv2.putText(frame_i, f"Mano 3D: {xm:.2f}, {ym:.2f}, {zm:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Camara Izquierda", frame_i)
        cv2.imshow("Camara Derecha", frame_d)
        if cv2.waitKey(1) == 27: break

    cam_izq.release()
    cam_der.release()
    cv2.destroyAllWindows()
