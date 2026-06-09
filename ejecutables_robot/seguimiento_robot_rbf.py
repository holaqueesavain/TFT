"""
Control del robot UR5e mediante seguimiento de mano en tiempo real.
Usa un modelo RBF entrenado sobre el grid de calibración para estimar
la posición del robot a partir de las coordenadas de la cámara.
"""

import rtde_control, rtde_receive
import cv2
import numpy as np
import csv, os, sys, time, glob, xmlrpc.client
from scipy.interpolate import Rbf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from modulos_vision.seguimiento_mano import (
    detector_manos_i, detector_manos_d, obtener_centro_mano,
    detectar_punta_verde, mano_cerrada, triangular
)
from ejecutables_robot.seguimiento_robot_lineal import limitar_paso

ROBOT_IP = "169.254.129.110"

def cargar_datos_mapeo():
    archivos = glob.glob(os.path.join(BASE_DIR, "output", "mapeo_*.csv"))
    if not archivos:
        print("ERROR: No hay mapeo CSV"); sys.exit(1)
    ultimo_archivo = max(archivos, key=os.path.getctime)
    cx, cy, cz, rx, ry, rz = [], [], [], [], [], []
    with open(ultimo_archivo, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for fila in reader:
            if len(fila) == 7 and fila[4] != "ERROR":
                rx.append(float(fila[1])); ry.append(float(fila[2])); rz.append(float(fila[3]))
                cx.append(float(fila[4])); cy.append(float(fila[5])); cz.append(float(fila[6]))
    return np.array(cx), np.array(cy), np.array(cz), np.array(rx), np.array(ry), np.array(rz)

CX, CY, CZ, RX, RY, RZ = cargar_datos_mapeo()


modelo_x = Rbf(CX, CY, CZ, RX, function='multiquadric', smooth=0.5)
modelo_y = Rbf(CX, CY, CZ, RY, function='multiquadric', smooth=0.5)
modelo_z = Rbf(CX, CY, CZ, RZ, function='multiquadric', smooth=0.5)

L_ROB_X = [np.min(RX), np.max(RX)]
L_ROB_Y = [np.min(RY), np.max(RY)]
L_ROB_Z = [np.min(RZ), np.max(RZ)]

OFFSET_Z = 0.05

def cam2robot(cam_x, cam_y, cam_z):
    rx_raw = modelo_x(cam_x, cam_y, cam_z)
    ry = modelo_y(cam_x, cam_y, cam_z)
    rz = modelo_z(cam_x, cam_y, cam_z)
    rx = L_ROB_X[0] + L_ROB_X[1] - rx_raw
    return [float(rx), float(ry), float(rz)]

if __name__ == "__main__":
    try:
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        gripper = xmlrpc.client.ServerProxy(f"http://{ROBOT_IP}:41414/")
        rot_fija   = [3.140198214586112, 0.0, 0.0]
        pose_inicio = [-0.11557, -0.4964, L_ROB_Z[0] + 0.1] + rot_fija
        rtde_c.moveL(pose_inicio, 0.1, 0.1)
    except Exception as e:
        print(f"Error conexión: {e}"); sys.exit(1)

    cam_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    for c in (cam_i, cam_d):
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    st_mano   = (0.0, 0.0, 0.0)
    st_robot  = (0.0, 0.0, 0.0)
    last_pose = pose_inicio[:]
    primer_contacto = True

    while True:
        ret_i, fi = cam_i.read()
        ret_d, fd = cam_d.read()
        if not ret_i or not ret_d: break

        fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)
        res_i = detector_manos_i.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos_d.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))

        p_i = p_d = lm = None
        if res_i.multi_hand_landmarks:
            lm = res_i.multi_hand_landmarks[0]
            p_i = obtener_centro_mano(lm.landmark)
            cv2.circle(fi, p_i, 8, (255, 0, 0), -1)
        if res_d.multi_hand_landmarks:
            p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark)

        r_i = detectar_punta_verde(fi, hand_landmarks=lm)
        r_d = detectar_punta_verde(fd)

        mano, st_mano   = triangular(p_i, p_d, 980, st_mano)
        robot, st_robot = triangular(r_i, r_d, 980, st_robot)

        if mano:
            if lm and mano_cerrada(lm.landmark) >= 3:
                rtde_c.servoStop(); primer_contacto = True
                cv2.putText(fi, "PAUSADO", (20, 80), 0, 0.8, (0, 0, 255), 2)
            else:
                rx_raw, ry_raw, rz_raw = cam2robot(*mano)
                rx = np.clip(rx_raw, -0.50, 0.25) 
                ry = np.clip(ry_raw, -0.85, -0.20)
                rz = np.clip(rz_raw, -0.25, 0.30)

                pose_obj = [rx, ry, rz] + rot_fija
                if primer_contacto: primer_contacto = False

                distancia = np.linalg.norm(np.array(pose_obj[:3]) - np.array(last_pose[:3]))
                if distancia > 0.003:
                    pose_suave = limitar_paso(last_pose[:3], pose_obj[:3], paso_max=0.008)
                    pose_final = list(pose_suave) + rot_fija
                    if rtde_c.getInverseKinematics(pose_final):
                        rtde_c.servoL(pose_final, 0.5, 0.1, 0.05, 0.15, 150)
                        last_pose = pose_final
                    else:
                        cv2.putText(fi, "IK ERROR", (20, 80), 0, 0.8, (0, 165, 255), 2)

            if r_i: cv2.arrowedLine(fi, r_i, p_i, (0, 255, 255), 3)
        else:
            cv2.putText(fi, "BUSCANDO MANO", (20, 50), 0, 0.65, (0, 0, 255), 2)

        cv2.imshow("Control UR5e", fi)
        cv2.imshow("Camara Derecha", fd)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27: break
        elif tecla == ord('r'):
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

    rtde_c.servoStop()
    rtde_c.moveL(pose_inicio, 0.1, 0.1)
    cam_i.release(); cam_d.release()
    rtde_c.disconnect(); rtde_r.disconnect()
    cv2.destroyAllWindows()