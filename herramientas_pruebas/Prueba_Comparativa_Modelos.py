"""
Herramienta de evaluación en tiempo real para los modelos Lineal, Matriz y RBF.
Registra en un CSV el error de posicionamiento y la carga computacional del sistema, 
permitiendo alternar entre los distintos modelos matemáticos durante la ejecución.
"""
import cv2
import sys, os
import numpy as np
import csv, time
import rtde_control, rtde_receive
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modulos_vision.seguimiento_mano import (
    detector_manos_i, detector_manos_d, obtener_centro_mano, triangular, detectar_punta_verde
)
from ejecutables_robot.seguimiento_robot_lineal import cam2robot as f_lineal, limitar_paso
from ejecutables_robot.seguimiento_robot_matriz import cam2robot as f_matriz
from ejecutables_robot.seguimiento_robot_rbf import cam2robot as f_rbf

ROBOT_IP = "169.254.129.110"

def main():
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    
    f_csv = open(f"log_pruebas_{int(time.time())}.csv", "w", newline="")
    writer = csv.writer(f_csv, delimiter=";")
    writer.writerow(["Modelo", "Cam_X_Robot", "Cam_Y_Robot", "Cam_Z_Robot", "Robot_Pred_X", "Robot_Pred_Y", "Robot_Pred_Z", "Robot_Real_X", "Robot_Real_Y", "Robot_Real_Z", "Tiempo_Ms"])

    modelo_actual = "LINEAL"
    modelo_func = f_lineal
    paso_actual = 0.008
    rot_fija = [3.140198214586112, 0.0, 0.0]
    
    last_pose_sent = [-0.115, -0.496, 0.1] + rot_fija
    primer_contacto = True 
    
    cam_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    st_mano = (0.0, 0.0, 0.0)
    st_robot = (0.0, 0.0, 0.0)

    grabando = False
    inicio_grabacion = 0
    duracion_max = 15 

    while True:
        ret_i, fi = cam_i.read()
        ret_d, fd = cam_d.read()
        if not ret_i or not ret_d: break
        
        fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)
        res_i = detector_manos_i.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos_d.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))
        
        p_i = obtener_centro_mano(res_i.multi_hand_landmarks[0].landmark) if res_i.multi_hand_landmarks else None
        p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark) if res_d.multi_hand_landmarks else None
        
        lm_i = res_i.multi_hand_landmarks[0] if res_i.multi_hand_landmarks else None
        r_i = detectar_punta_verde(fi, hand_landmarks=lm_i)
        r_d = detectar_punta_verde(fd)

        mano, st_mano = triangular(p_i, p_d, 200, st_mano)
        robot_cam, st_robot = triangular(r_i, r_d, 300, st_robot)

        if mano:
            t0 = time.perf_counter()
            rx, ry, rz = modelo_func(*mano)
            tiempo_ms = (time.perf_counter() - t0) * 1000
            
            if grabando:
                tiempo_actual = time.time()
                if tiempo_actual - inicio_grabacion < duracion_max:
                    if robot_cam:
                        rx_pred, ry_pred, rz_pred = modelo_func(*robot_cam)
                        rz_pred -= 0.05
                        pos_real = rtde_r.getActualTCPPose()
                        
                        writer.writerow([modelo_actual, robot_cam[0], robot_cam[1], robot_cam[2], rx_pred, ry_pred, rz_pred, pos_real[0], pos_real[1], pos_real[2], tiempo_ms])
                    
                    cv2.putText(fi, f"GRABANDO: {duracion_max - int(tiempo_actual - inicio_grabacion)}s", (20, 100), 0, 1, (0, 0, 255), 2)
                else:
                    grabando = False

            rx = np.clip(rx, -0.50, 0.25)
            ry = np.clip(ry, -0.85, -0.20)
            rz = np.clip(rz, -0.25, 0.40)
            
            pose_objetivo = [rx, ry, rz] + rot_fija
            if primer_contacto:
                primer_contacto = False
                
            pose_suave_xyz = limitar_paso(last_pose_sent[:3], pose_objetivo[:3], paso_max=paso_actual)
            pose_final = list(pose_suave_xyz) + rot_fija
            
            dist_mov = np.linalg.norm(np.array(pose_final[:3]) - np.array(last_pose_sent[:3])) 
            if dist_mov > 0.0005:
                if rtde_c.getInverseKinematics(pose_final):
                    rtde_c.servoL(pose_final, 0.5, 0.1, 0.05, 0.1, 300)
                    last_pose_sent = pose_final
            
            cv2.circle(fi, p_i, 8, (255, 0, 0), -1)
            cv2.putText(fi, f"MODELO: {modelo_actual} ({tiempo_ms:.2f}ms)", (20, 50), 0, 1, (0, 255, 0), 2)
        else:
            primer_contacto = True 

        cv2.imshow("Camara Izquierda", fi)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in [ord('1'), ord('2'), ord('3')]:
            opciones = {
                "1": ("LINEAL", f_lineal, 0.008),
                "2": ("MATRIZ", f_matriz, 0.008),
                "3": ("RBF", f_rbf, 0.008)
            }
            modelo_actual, modelo_func, paso_actual = opciones[chr(tecla)]
            grabando, inicio_grabacion = True, time.time()
            primer_contacto = True
        elif tecla == 27: break

    f_csv.close()
    rtde_c.servoStop()
    cv2.destroyAllWindows()
    cam_i.release()
    cam_d.release()

if __name__ == "__main__":
    main()