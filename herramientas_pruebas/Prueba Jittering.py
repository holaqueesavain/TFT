"""
Prueba para ver cómo afecta el filtrado al ruido en la posición 3D de la mano.
"""

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
    
    vec = np.dot(Q, np.array([640 - p_izq[0], p_izq[1], disp, 1.0]))
    X, Y, Z = vec[0]/vec[3], vec[1]/vec[3], abs(vec[2]/vec[3])
    
    return (alpha * X + (1 - alpha) * anterior[0],
            alpha * Y + (1 - alpha) * anterior[1],
            alpha * Z + (1 - alpha) * anterior[2])

def ejecutar_prueba():
    cap_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_i.set(3, 640); cap_i.set(4, 480)
    cap_d.set(3, 640); cap_d.set(4, 480)

    if not cap_i.isOpened() or not cap_d.isOpened(): return

    estados = {
        'Alpha 1.0 (Sin filtro)': (0.0,0.0,0.0), 
        'Alpha 0.9 (Suave)': (0.0,0.0,0.0),
        'Alpha 0.4 (Medio)': (0.0,0.0,0.0), 
        'Alpha 0.1 (Fuerte)': (0.0,0.0,0.0) 
    }
    buffer = {k: [] for k in estados.keys()}

    while True:
        ret_i, fi = cap_i.read()
        ret_d, fd = cap_d.read()
        if not ret_i or not ret_d: break
        
        fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)
        res_i = detector_manos.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))
        
        p_i = obtener_centro_mano(res_i.multi_hand_landmarks[0].landmark) if res_i.multi_hand_landmarks else None
        p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark) if res_d.multi_hand_landmarks else None

        estados['Alpha 1.0 (Sin filtro)'] = triangulacion_estudio(p_i, p_d, 1.0, estados['Alpha 1.0 (Sin filtro)'])
        estados['Alpha 0.9 (Suave)'] = triangulacion_estudio(p_i, p_d, 0.9, estados['Alpha 0.9 (Suave)'])
        estados['Alpha 0.4 (Medio)'] = triangulacion_estudio(p_i, p_d, 0.4, estados['Alpha 0.4 (Medio)'])
        estados['Alpha 0.1 (Fuerte)'] = triangulacion_estudio(p_i, p_d, 0.1, estados['Alpha 0.1 (Fuerte)'])

        y_pos = 50
        for key in estados.keys():
            buffer[key].append(estados[key])
            if len(buffer[key]) > 20: buffer[key].pop(0)
            
            jitter = np.std([np.linalg.norm(p) for p in buffer[key]]) * 100 if len(buffer[key]) > 5 else 0
            
            color = (255,255,255)
            if '1.0' in key: color = (0,0,255)
            if '0.1' in key: color = (0,255,0)

            cv2.putText(fi, f"{key} - Jitter: {jitter:.3f} cm", (20, y_pos), 0, 0.5, color, 2)
            y_pos += 40

        cv2.imshow("Estudio de Estabilidad (Jittering)", fi)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap_i.release(); cap_d.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ejecutar_prueba()