import cv2
import sys, os
import numpy as np
import time

# Añadir el directorio raíz al path para poder importar los módulos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from modulos_vision.seguimiento_mano import (
    detector_manos_i, detector_manos_d, obtener_centro_mano, triangular
)

# Importar las funciones de transformación de los 3 modelos
from ejecutables_robot.seguimiento_robot_lineal import cam2robot as f_lineal
from ejecutables_robot.seguimiento_robot_matriz import cam2robot as f_matriz
from ejecutables_robot.seguimiento_robot_rbf import cam2robot as f_rbf

import rtde_receive

ROBOT_IP = "169.254.129.110"

def calcular_error(pos_predicha, pos_real):
    if pos_real is None: return 0.0
    # Distancia Euclidiana en 3D (X, Y, Z) en milímetros
    v1 = np.array(pos_predicha[:3])
    v2 = np.array(pos_real[:3])
    distancia_m = np.linalg.norm(v1 - v2)
    return distancia_m * 1000.0  # Convertir a mm

def main():
    print("Iniciando Prueba Comparativa de Modelos...")
    
    # Intentar conectar al robot en MODO LECTURA
    rtde_r = None
    try:
        print("Conectando al robot para leer posicion real (Ground Truth)...")
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        print("Conexion al robot: OK")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo conectar al robot ({e}). La precision no se calculara.")
    
    cam_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Bajar resolución para evitar saturar el bus USB / Bluetooth
    for cam in (cam_i, cam_d):
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)

    st_mano = (0.0, 0.0, 0.0)

    while True:
        ret_i, fi = cam_i.read()
        ret_d, fd = cam_d.read()
        if not ret_i or not ret_d: 
            print("Error al leer las camaras.")
            break
        
        # fi, fd = cv2.flip(fi, 1), cv2.flip(fd, 1) # Efecto espejo desactivado

        res_i = detector_manos_i.process(cv2.cvtColor(fi, cv2.COLOR_BGR2RGB))
        res_d = detector_manos_d.process(cv2.cvtColor(fd, cv2.COLOR_BGR2RGB))
        
        p_i = p_d = None
        if res_i.multi_hand_landmarks:
            p_i = obtener_centro_mano(res_i.multi_hand_landmarks[0].landmark)
            cv2.circle(fi, p_i, 8, (255, 0, 0), -1)
            
        if res_d.multi_hand_landmarks:
            p_d = obtener_centro_mano(res_d.multi_hand_landmarks[0].landmark)

        # Triangular posición de la mano
        mano, st_mano = triangular(p_i, p_d, 200, st_mano)

        if mano:
            xm, ym, zm = mano
            
            # Leer posición real del robot si está conectado
            pos_real = None
            if rtde_r:
                pos_real = rtde_r.getActualTCPPose()
            
            # --- MODELO LINEAL ---
            t0 = time.perf_counter()
            pos_lineal = f_lineal(xm, ym, zm)
            t_lineal_ms = (time.perf_counter() - t0) * 1000
            err_lineal = calcular_error(pos_lineal, pos_real)
            
            # --- MODELO MATRIZ ---
            t0 = time.perf_counter()
            pos_matriz = f_matriz(xm, ym, zm)
            t_matriz_ms = (time.perf_counter() - t0) * 1000
            err_matriz = calcular_error(pos_matriz, pos_real)
            
            # --- MODELO RBF ---
            t0 = time.perf_counter()
            pos_rbf = f_rbf(xm, ym, zm)
            t_rbf_ms = (time.perf_counter() - t0) * 1000
            err_rbf = calcular_error(pos_rbf, pos_real)
            
            # Formatear textos
            if pos_real:
                texto_robot = f"Robot Real: ({pos_real[0]:.3f}, {pos_real[1]:.3f}, {pos_real[2]:.3f})"
                cv2.putText(fi, texto_robot, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            texto_lin = f"LIN: ({pos_lineal[0]:.3f}, {pos_lineal[1]:.3f}, {pos_lineal[2]:.3f}) | {t_lineal_ms:.2f}ms | Err: {err_lineal:.1f}mm"
            texto_mat = f"MAT: ({pos_matriz[0]:.3f}, {pos_matriz[1]:.3f}, {pos_matriz[2]:.3f}) | {t_matriz_ms:.2f}ms | Err: {err_matriz:.1f}mm"
            texto_rbf = f"RBF: ({pos_rbf[0]:.3f}, {pos_rbf[1]:.3f}, {pos_rbf[2]:.3f}) | {t_rbf_ms:.2f}ms | Err: {err_rbf:.1f}mm"
            
            # Imprimir en consola de vez en cuando
            if np.random.rand() > 0.9:
                print(f"Cam({xm:.2f},{ym:.2f},{zm:.2f}) -> L:{err_lineal:.1f}mm, M:{err_matriz:.1f}mm, R:{err_rbf:.1f}mm")
            
            # Mostrar en la pantalla de OpenCV
            cv2.putText(fi, texto_lin, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(fi, texto_mat, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(fi, texto_rbf, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        else:
            cv2.putText(fi, "BUSCANDO MANO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Camara Izquierda (Comparacion)", fi)
        cv2.imshow("Camara Derecha", fd)

        # Aumentamos el delay a 30ms (aprox 33 FPS máximos) para relajar la CPU
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == 27: # ESC para salir
            break

    cam_i.release()
    cam_d.release()
    if rtde_r: rtde_r.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
