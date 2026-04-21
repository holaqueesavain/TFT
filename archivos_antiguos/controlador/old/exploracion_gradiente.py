import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

"""
Este codigo implementa un sistema de gradiente 
descendente para mover el robot. TEST 
"""

# ROBOT_IP = "192.168.19.128"  # simulador
ROBOT_IP = "169.254.129.110"  # robot real

Z_PLANO = -0.277
PASO_INICIAL = 0.06      # 6 cm
PASO_MIN = 0.005         # 0.5 cm
MAX_PASOS = 60
TOL_PIXEL = 20
DELAY_LECTURA = 0.25

ESCALA_PX_A_M = 0.0018

LIMITE_X_MIN = -0.390
LIMITE_X_MAX = 0.160
LIMITE_Y_MIN = -0.815 
LIMITE_Y_MAX = -0.265

def cargar_mallado(filepath="C:\\ARIS\\universidad\\TFG\\Programa\\sistemavision\\output\\puntos_mallado.txt"):
    puntos = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo {filepath} no encontrado")
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                x, y = map(float, line.strip().split())
                puntos[f"p{i}"] = [x, y]
    return puntos

def leer_punta_robot(filepath="C:\\ARIS\\universidad\\TFG\\Programa\\sistemavision\\output\\punta_robot.txt"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        line = f.readline().strip()
        if line == "None" or not line:
            return None
        try:
            return list(map(float, line.split()))
        except:
            return None

def distancia_2d(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

rotacion_abajo = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec().tolist()

puntos_mallado = cargar_mallado()
print(f"Cargados {len(puntos_mallado)} puntos del mallado")

pos_inicial = [-0.11947692474323461, -0.270, -0.2773488558014402, 3.140198214586112, 0.0064768660533132535, 0.0062702868250280284]
memoria = {}

def buscar_punto_cercano_no_visitado(punta, puntos_mallado, memoria, umbral=12):
    for pk, pv in puntos_mallado.items():
        if pk in memoria:
            continue
        if distancia_2d(punta, pv) <= umbral:
            return pk, pv
    return None, None

for target_key in sorted(puntos_mallado.keys(), key=lambda x: int(x[1:])):  # p0, p1, p2, ...
    if target_key in memoria:
        print(f"\n{target_key} ya alcanzado. Saltando.")
        continue

    print("Moviendo a posición inicial antes de explorar...")
    rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)
    pose_actual = rtde_r.getActualTCPPose()

    target_2d = puntos_mallado[target_key]
    print(f"\n{'='*60}")
    print(f"INICIANDO navegación directa para: {target_key} = ({target_2d[0]:.1f}, {target_2d[1]:.1f})")
    print(f"{'='*60}")

    paso = 0

    try:
        while paso < MAX_PASOS:
            punta_actual = leer_punta_robot()
            if punta_actual is None:
                print("No se detecta la punta roja. Esperando...")
                time.sleep(DELAY_LECTURA)
                continue

            dist_actual = distancia_2d(punta_actual, target_2d)
            print(f"\n--- Paso {paso + 1} ---")
            print(f"Distancia a {target_key}: {dist_actual:.1f} px")

            # aceptar objetivo si está suficientemente cerca
            if dist_actual < TOL_PIXEL: 
                print(f"¡Objetivo alcanzado (distancia: {dist_actual:.1f} px)!")
                memoria[target_key] = rtde_r.getActualTCPPose()
                rtde_c.moveL(memoria[target_key], speed=0.08, acceleration=0.08)
                print("Esperando 5 segundos en el objetivo...")
                time.sleep(5)
                break

            punto_cercano_key, _ = buscar_punto_cercano_no_visitado(
                punta_actual, puntos_mallado, memoria, umbral=10
            )
            if punto_cercano_key is not None and punto_cercano_key != target_key:
                print(f"\n¡Detectado punto cercano no visitado: {punto_cercano_key}!")
                memoria[punto_cercano_key] = rtde_r.getActualTCPPose()
                rtde_c.moveL(memoria[punto_cercano_key], speed=0.1, acceleration=0.1)
                print("Esperando 5 segundos en el punto cercano...")
                time.sleep(5)
                pose_actual = rtde_r.getActualTCPPose()
                continue

            dx_px = target_2d[0] - punta_actual[0]
            dy_px = target_2d[1] - punta_actual[1]
            distancia_px = np.hypot(dx_px, dy_px)

            if distancia_px == 0:
                print("Posición objetivo idéntica. Terminando.")
                break

            dx_norm = dx_px / distancia_px
            dy_norm = dy_px / distancia_px

            # calcular un paso adaptativo
            if dist_actual < 20:
                paso_m = 0.01   # 8 mm
            elif dist_actual < 35:
                paso_m = 0.02   # 12 mm
            elif dist_actual < 60:
                paso_m = 0.06   # 20 mm
            else:
                paso_m = min(PASO_INICIAL, dist_actual * ESCALA_PX_A_M * 0.8)

            x_new = pose_actual[0] + dx_norm * paso_m
            y_new = pose_actual[1] + dy_norm * paso_m

            x_new = np.clip(x_new, LIMITE_X_MIN, LIMITE_X_MAX)
            y_new = np.clip(y_new, LIMITE_Y_MIN, LIMITE_Y_MAX)

            nueva_pose = [x_new, y_new, Z_PLANO] + rotacion_abajo
            rtde_c.moveL(nueva_pose, speed=0.15, acceleration=0.15)
            time.sleep(DELAY_LECTURA)

            pose_actual = rtde_r.getActualTCPPose()
            paso += 1

        else:
            print(f"\nLímite de {MAX_PASOS} pasos alcanzado para {target_key}.")
    except Exception as e:
        print(f"Error durante la navegación de {target_key}: {e}")

print("\nVolviendo a posición inicial final...")
rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)

try:
    rtde_c.disconnect()
    rtde_r.disconnect()
except:
    pass

print(f"\n{'='*60}")
print("RESULTADO FINAL")
print(f"{'='*60}")
if memoria:
    print(f"Puntos alcanzados: {len(memoria)}/{len(puntos_mallado)}")
    print("\n=== MEMORIA COMPLETA ===")
    for k in sorted(memoria.keys(), key=lambda x: int(x[1:])):
        v = memoria[k]
        print(f"{k}: x={v[0]:.4f}, y={v[1]:.4f}")
else:
    print("Ningún punto fue alcanzado.")

print("\nPrograma terminado.")