import rtde_control, rtde_receive
import cv2
import numpy as np
import csv, os, sys, time, datetime

# Configuración de rutas para encontrar modulos_vision y ejecutables_robot
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from modulos_vision.seguimiento_mano import detector_manos, obtener_centro_mano, triangular, mano_cerrada
from ejecutables_robot.seguimiento_robot import cam2robot, ROBOT_IP, L_ROB_Z, limitar_paso, L_ROB_X, L_ROB_Y

rot_fija = [3.140198214586112, 0.0064768660533132535, 0.0062702868250280284]

def prueba_seguimiento_real():
    print("\n" + "="*50)
    print(">> TEST DE SEGUIMIENTO DINAMICO REAL")
    print("="*50)
    print("INSTRUCCIONES:")
    print("1. El robot se movera para seguirte.")
    print("2. Pulsa 'G' para empezar a grabar el movimiento.")
    print("3. Pulsa 'G' otra vez para parar y generar el Excel.")
    print("4. Pulsa 'ESC' para salir.")
    print("="*50)

    try:
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        pose_inicio = [-0.118, -0.513, L_ROB_Z[0]] + rot_fija
        rtde_c.moveL(pose_inicio, 0.1, 0.1)
        print("OK: Robot conectado y en posicion de inicio.")
    except Exception as e:
        print(f"Error de conexion robot: {e}")
        return

    cap_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    st_mano = (0.0, 0.0, 0.0)
    last_pose_sent = pose_inicio
    primer_contacto = True
    
    grabando = False
    datos_capturados = []

    while True:
        ret_i, fi = cap_i.read()
        ret_d, fd = cap_d.read()
        if not ret_i or not ret_d: break
        fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1)

        res_i = detector_manos.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))
        
        lm = p_i = p_d = None
        if res_i.multi_hand_landmarks:
            lm = res_i.multi_hand_landmarks[0]
            p_i = obtener_centro_mano(lm.landmark)
        if res_d.multi_hand_landmarks:
            p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark)

        mano, st_mano = triangular(p_i, p_d, 200, st_mano)

        if mano:
            xm, ym, zm = mano
            rx_raw, ry_raw, rz_raw = cam2robot(xm, ym, zm)
            
            # Limites de seguridad
            rx = np.clip(rx_raw, np.min(L_ROB_X), np.max(L_ROB_X))
            ry = np.clip(ry_raw, np.min(L_ROB_Y), np.max(L_ROB_Y))
            rz = np.clip(rz_raw, L_ROB_Z[0], L_ROB_Z[1]) 
            pose_obj = [rx, ry, rz] + rot_fija

            # Movimiento del robot (Igual que en el principal)
            if not (lm and mano_cerrada(lm.landmark) >= 3):
                if primer_contacto:
                    rtde_c.moveL(pose_obj, 0.1, 0.05)
                    last_pose_sent = pose_obj
                    primer_contacto = False
                else:
                    pose_suave_xyz = limitar_paso(last_pose_sent[:3], pose_obj[:3], paso_max=0.02)
                    pose_obj_final = list(pose_suave_xyz) + rot_fija
                    if np.linalg.norm(np.array(pose_obj_final[:3]) - np.array(last_pose_sent[:3])) > 0.0005:
                        if rtde_c.getInverseKinematics(pose_obj_final):
                            rtde_c.servoL(pose_obj_final, 0.5, 0.1, 0.05, 0.1, 300)
                            last_pose_sent = pose_obj_final

            # LOGICA DE GRABACION
            if grabando:
                actual_tcp = rtde_r.getActualTCPPose()
                tiempo_actual = time.time() - tiempo_inicio
                datos_capturados.append({
                    'Tiempo': round(tiempo_actual, 3),
                    'Vis_X': rx, 'Vis_Y': ry, 'Vis_Z': rz,
                    'Rob_Real_X': actual_tcp[0], 'Rob_Real_Y': actual_tcp[1], 'Rob_Real_Z': actual_tcp[2]
                })
                cv2.circle(fi, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(fi, f"GRABANDO: {len(datos_capturados)}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(fi, "PULSA 'G' PARA GRABAR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("TEST DINAMICO REAL", fi)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27: break
        elif tecla == ord('g'):
            if not grabando:
                grabando = True
                datos_capturados = []
                tiempo_inicio = time.time()
                print(">> Grabacion iniciada...")
            else:
                grabando = False
                print(f">> Grabacion finalizada. {len(datos_capturados)} muestras.")
                if datos_capturados:
                    nombre = f"datos_seg_real_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    ruta = os.path.join(BASE_DIR, "output", nombre)
                    with open(ruta, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=datos_capturados[0].keys(), delimiter=';')
                        writer.writeheader()
                        writer.writerows(datos_capturados)
                    print(f">> Archivo guardado: {ruta}")

    rtde_c.servoStop()
    rtde_c.moveL(pose_inicio, 0.1, 0.1)
    rtde_c.disconnect()
    rtde_r.disconnect()
    cap_i.release()
    cap_d.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    prueba_seguimiento_real()
