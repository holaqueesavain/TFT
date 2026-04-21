import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
detector_manos = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.2, 
    min_tracking_confidence=0.3,  
    max_num_hands=2
)

# Calibracion
BASELINE = 0.093      
FOCAL = 728.0
CX = 320.0
CY = 240.0

Q = np.float32([
    [1, 0, 0, -CX],
    [0, 1, 0, -CY],
    [0, 0, 0, FOCAL],
    [0, 0, -1.0/BASELINE, 0]
])

ALPHA = 0.40

def mano_cerrada(landmarks):
    puntas = [8, 12, 16, 20]
    nudillos = [6, 10, 14, 18]
    dedos_doblados = 0
    px_base, py_base = landmarks[0].x, landmarks[0].y
    for i in range(4):
        dist_punta = (landmarks[puntas[i]].x - px_base)**2 + (landmarks[puntas[i]].y - py_base)**2
        dist_nudillo = (landmarks[nudillos[i]].x - px_base)**2 + (landmarks[nudillos[i]].y - py_base)**2
        if dist_punta < dist_nudillo:
            dedos_doblados += 1
    return dedos_doblados

def obtener_centro_mano(landmarks):
    x = int((landmarks[0].x*640 + landmarks[5].x*640 + landmarks[17].x*640) / 3)
    y = int((landmarks[0].y*480 + landmarks[5].y*480 + landmarks[17].y*480) / 3)
    return np.clip(x, 0, 639), np.clip(y, 0, 479)

def detectar_punta_verde(frame, hand_landmarks=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([38, 18, 107])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 10:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                return np.clip(cx, 0, 639), np.clip(cy, 0, 479)
    return None

def triangular(p_izq, p_der, tolerancia_y, estado_anterior=None):
    if p_izq and p_der:
        if abs(p_izq[1] - p_der[1]) < tolerancia_y:
            disp = (640 - p_izq[0]) - (640 - p_der[0])
            if abs(disp) > 1:
                vec = np.dot(Q, np.array([640 - p_izq[0], p_izq[1], disp, 1.0]))
                if vec[3] != 0:
                    X, Y, Z = vec[0]/vec[3], vec[1]/vec[3], abs(vec[2]/vec[3])
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
    estado_mano = estado_robot = (0, 0, 0)
    
    L_X, L_Z = [-2.0, 2.0], [0.0, 3.0]
    OFFSET_Y = 0.30

    print("--- MOTOR DE VISIÓN V2 ---")

    while True:
        ret_i, frame_i = cam_izq.read()
        ret_d, frame_d = cam_der.read()
        if not ret_i or not ret_d: break

        frame_i, frame_d = cv2.flip(frame_i, 1), cv2.flip(frame_d, 1)
        res_i = detector_manos.process(cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB))
        res_d = detector_manos.process(cv2.cvtColor(frame_d, cv2.COLOR_BGR2RGB))

        p_i = obtener_centro_mano(res_i.multi_hand_landmarks[0].landmark) if res_i.multi_hand_landmarks else None
        p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark) if res_d.multi_hand_landmarks and len(res_d.multi_hand_landmarks)==1 else None
        r_i = detectar_punta_verde(frame_i, hand_landmarks=res_i.multi_hand_landmarks[0] if res_i.multi_hand_landmarks else None)
        r_d = detectar_punta_verde(frame_d)

        mano_3d, estado_mano = triangular(p_i, p_d, 200, estado_mano)
        robot_3d, estado_robot = triangular(r_i, r_d, 300, estado_robot)

        if p_i: cv2.circle(frame_i, p_i, 5, (255,0,0), -1)
        if r_i: cv2.circle(frame_i, r_i, 6, (0,255,0), -1)

        if mano_3d:
            xm, ym, zm = mano_3d
            color = (0, 255, 0)
            if xm < L_X[0] or xm > L_X[1] or zm < L_Z[0] or zm > L_Z[1]:
                color = (0, 165, 255)
                cv2.putText(frame_i, "LIMITE ALCANZADO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
        if robot_3d:
            xr, yr, zr = robot_3d
            print(f"Robot Z: {zr * 100:.1f} cm")
            if mano_3d:
                xm, ym, zm = mano_3d
                error = (xm - xr, (ym - OFFSET_Y) - yr, zm - zr)
                cv2.arrowedLine(frame_i, r_i, p_i, (0, 255, 255), 3)
                cv2.putText(frame_i, f"Err X:{error[0] * 100:.2f} cm", (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 3)
                cv2.putText(frame_i, f"Err Y:{error[1] * 100:.2f} cm", (20,240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 3)
                cv2.putText(frame_i, f"Err Z:{error[2] * 100:.2f} cm", (20,280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 3)

        cv2.imshow("Vision - Camara Izquierda", frame_i)
        cv2.imshow("Vision - Camara Derecha", frame_d)
        if cv2.waitKey(1) == 27: break

    cam_izq.release()
    cam_der.release()
    cv2.destroyAllWindows()