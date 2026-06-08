"""
Control del robot UR5e mediante seguimiento de mano en tiempo real.
Usa una Matriz de Transformación Homogénea sobre los puntos de calibración para convertir
las coordenadas de la cámara al sistema de referencia del robot.
"""

import rtde_control, rtde_receive
import cv2
import numpy as np
import csv, os, sys, time, glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from modulos_vision.seguimiento_mano import (
    detector_manos_i, detector_manos_d, obtener_centro_mano,
    detectar_punta_verde, mano_cerrada, triangular
)
from ejecutables_robot.seguimiento_robot_lineal import limitar_paso

ROBOT_IP = "169.254.129.110"

LIMITE_ROB_X = [-0.34, 0.11] 
LIMITE_ROB_Y = [-0.71, -0.31] 
LIMITE_ROB_Z = [-0.18, 0.20] 

def calcular_matriz_transformacion():
    archivos = glob.glob(os.path.join(BASE_DIR, "output", "mapeo_*.csv"))
    if not archivos:
        print("ERROR: No hay mapeo CSV"); sys.exit(1)
    ultimo_archivo = max(archivos, key=os.path.getctime)
    
    columnas_camara = []
    columnas_robot = []
    
    with open(ultimo_archivo, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) 
        for fila in reader:
            if fila[4] == "ERROR":
                continue
                
            columnas_robot.append([float(fila[1]), float(fila[2]), float(fila[3])])
            columnas_camara.append([float(fila[4]), float(fila[5]), float(fila[6])])

    C = np.array(columnas_camara)
    R = np.array(columnas_robot)

    c_mean = np.mean(C, axis=0)
    r_mean = np.mean(R, axis=0)

    C_centered = C - c_mean
    R_centered = R - r_mean

    H = C_centered.T @ R_centered

    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    Rot = V @ U.T

    t = r_mean - Rot @ c_mean

    T = np.eye(4)
    T[0:3, 0:3] = Rot
    T[0:3, 3] = t
    return T

MATRIZ_TRANSFORMACION = calcular_matriz_transformacion()

OFFSET_Z = 0.02

def cam2robot(cam_x, cam_y, cam_z):
    vector_cam = np.array([cam_x, cam_y, cam_z, 1.0])
    vector_rob = MATRIZ_TRANSFORMACION @ vector_cam

    rx = np.clip(vector_rob[0], LIMITE_ROB_X[0], LIMITE_ROB_X[1])
    ry = np.clip(vector_rob[1], LIMITE_ROB_Y[0], LIMITE_ROB_Y[1])
    rz_raw = np.clip(vector_rob[2] + OFFSET_Z, LIMITE_ROB_Z[0], LIMITE_ROB_Z[1])
    
    # Inversión de Z solicitada (igual que en RBF):
    rz = LIMITE_ROB_Z[0] + LIMITE_ROB_Z[1] - rz_raw
    
    return [float(rx), float(ry), float(rz)]

if __name__ == "__main__":
    print("Iniciando conexion robot (METODO MATRIZ)...")
    try:
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        rot_fija = [3.140198214586112, 0.0, 0.0]
        pose_inicio = [-0.11557, -0.4964, LIMITE_ROB_Z[0] + 0.1] + rot_fija
        
        rtde_c.moveL(pose_inicio, 0.1, 0.1)
    except Exception as e:
        print(f"Error de conexion: {e}")
        sys.exit(1)

    cam_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    st_mano = st_robot = (0.0, 0.0, 0.0)
    last_pose_sent = pose_inicio
    primer_contacto = True
    
    while True:
        ret_i, fi = cam_i.read()
        ret_d, fd = cam_d.read()
        if not ret_i or not ret_d: break
        
        fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)

        res_i = detector_manos_i.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos_d.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))
        
        lm = p_i = p_d = None
        if res_i.multi_hand_landmarks:
            lm = res_i.multi_hand_landmarks[0]
            p_i = obtener_centro_mano(lm.landmark)
            cv2.circle(fi, p_i, 8, (255, 0, 0), -1)
            
        if res_d.multi_hand_landmarks:
            p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark)
            
        r_i = detectar_punta_verde(fi, hand_landmarks=lm)
        r_d = detectar_punta_verde(fd)

        mano, st_mano = triangular(p_i, p_d, 200, st_mano)
        robot, st_robot = triangular(r_i, r_d, 300, st_robot)

        if mano:
            if lm and mano_cerrada(lm.landmark) >= 3:
                rtde_c.servoStop() 
                cv2.putText(fi, "PAUSADO", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                primer_contacto = True 
            else:
                xm, ym, zm = mano
                rx, ry, rz = cam2robot(xm, ym, zm)
                
                pose_obj = [rx, ry, rz] + rot_fija
                
                if primer_contacto: 
                    primer_contacto = False

                pose_suave_xyz = limitar_paso(last_pose_sent[:3], pose_obj[:3], paso_max=0.008)
                pose_obj_final = list(pose_suave_xyz) + rot_fija
                dist_mov = np.linalg.norm(np.array(pose_obj_final[:3]) - np.array(last_pose_sent[:3])) 
                
                if dist_mov > 0.0005: 
                    if rtde_c.getInverseKinematics(pose_obj_final):
                        rtde_c.servoL(pose_obj_final, 0.5, 0.1, 0.05, 0.1, 300)
                        last_pose_sent = pose_obj_final
                    else:
                        cv2.putText(fi, "IK ERROR", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    cv2.putText(fi, "ESTABLE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(fi, f"Robot: ({rx:.3f}, {ry:.3f}, {rz:.3f})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if r_i:
                cv2.arrowedLine(fi, r_i, p_i, (0, 255, 255), 3)
        else:
            cv2.putText(fi, "BUSCANDO MANO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


        cv2.imshow("Camara Izquierda", fi)
        cv2.imshow("Camara Derecha", fd)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        elif tecla == ord('c'):
            print("Recoger Pelota")
            rtde_c.servoStop()
            rtde_c.moveL([-0.2456, -0.821, -0.385] + rot_fija, 0.1, 0.1)
            rtde_c.setStandardDigitalOut(0, True)
            time.sleep(1)
            rtde_c.moveL([-0.10, -0.5, 0.1] + rot_fija, 0.1, 0.1)
            primer_contacto = True
        elif tecla == ord('s'):
            print("Soltar Pelota")
            rtde_c.setStandardDigitalOut(0, False)
            time.sleep(1)

    try:
        rtde_c.servoStop()
        rtde_c.moveL(pose_inicio, 0.1, 0.1)
    except: pass

    cam_i.release()
    cam_d.release()
    rtde_c.disconnect()
    rtde_r.disconnect()
    cv2.destroyAllWindows()
