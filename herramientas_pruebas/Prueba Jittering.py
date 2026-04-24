import cv2
import numpy as np
import os, sys, time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from modulos_vision.seguimiento_mano import detector_manos, obtener_centro_mano, Q

def triangulacion_estudio(p_izq, p_der, alpha, anterior):
    if not (p_izq and p_der): return anterior
    if abs(p_izq[1] - p_der[1]) > 200: return anterior
    disp = (640 - p_izq[0]) - (640 - p_der[0])
    if abs(disp) <= 1: return anterior
    vec = np.dot(Q, np.array([640 - p_izq[0], p_izq[1], disparidad, 1.0])) if 'disparidad' in locals() else np.dot(Q, np.array([640 - p_izq[0], p_izq[1], disp, 1.0]))
    X, Y, Z = vec[0]/vec[3], vec[1]/vec[3], abs(vec[2]/vec[3])
    return (alpha * X + (1 - alpha) * anterior[0],
            alpha * Y + (1 - alpha) * anterior[1],
            alpha * Z + (1 - alpha) * anterior[2])

def prueba_jittering_alumno():
    cap_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_i.set(3, 640); cap_i.set(4, 480)
    cap_d.set(3, 640); cap_d.set(4, 480)

    if not cap_i.isOpened() or not cap_d.isOpened(): return

    # Definimos los estados para la comparativa técnica
    st = {
        'Alpha 1.0': (0.0,0.0,0.0), 
        'Alpha 0.9': (0.0,0.0,0.0),
        'Alpha 0.4': (0.0,0.0,0.0), 
        'Alpha 0.1': (0.0,0.0,0.0) 
    }
    buffer = {k: [] for k in st.keys()}

    while True:
        ret_i, fi = cap_i.read()
        ret_d, fd = cap_d.read()
        if not ret_i or not ret_d: break
        
        fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)
        res_i = detector_manos.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))
        
        p_i = obtener_centro_mano(res_i.multi_hand_landmarks[0].landmark) if res_i.multi_hand_landmarks else None
        p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark) if res_d.multi_hand_landmarks else None

        st['Alpha 1.0'] = triangulacion_estudio(p_i, p_d, 1.0, st['Alpha 1.0'])
        st['Alpha 0.9'] = triangulacion_estudio(p_i, p_d, 0.9, st['Alpha 0.9'])
        st['Alpha 0.4'] = triangulacion_estudio(p_i, p_d, 0.4, st['Alpha 0.4'])
        st['Alpha 0.1'] = triangulacion_estudio(p_i, p_d, 0.1, st['Alpha 0.1'])

        y_pos = 50
        for key in st.keys():
            buffer[key].append(st[key])
            if len(buffer[key]) > 20: buffer[key].pop(0)
            
            jitter = np.std([np.linalg.norm(p) for p in buffer[key]]) * 100 if len(buffer[key]) > 5 else 0
            
            # Colores neutros (Verde para el seleccionado, Blanco para el resto)
            color = (0,255,0) if '0.4' in key else (255,255,255)
            # Rojo para el RAW por ser el caso base
            if '1.0' in key: color = (0,0,255)

            cv2.putText(fi, f"{key} - Jitter: {jitter:.3f} cm", (20, y_pos), 0, 0.5, color, 2)
            y_pos += 40

        cv2.imshow("Prueba", fi)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap_i.release(); cap_d.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    prueba_jittering_alumno()