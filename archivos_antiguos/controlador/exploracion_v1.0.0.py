import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

"""
Este codigo realiza una exploracion del area
de trabajo moviendo el robot segun la informacion
proporcionada por la camara. 
"""

# para medir tiempo de ejecucion
inicio_ejecucion = time.time()

ROBOT_IP = "192.168.23.128"  # hiper-escaneada
#ROBOT_IP = "169.254.129.110"  # robot real

Z_PLANO = -0.277
PASO_INICIAL = 0.06      # 6 cm
PASO_MIN = 0.01          # 1 cm
MAX_FALLAS = 3
MAX_PASOS = 50
TOL_PIXEL = 18           # valor general
DELAY_LECTURA = 0.6
MEJORA_MINIMA = 3        # píxeles
TIEMPO_MAX_SIN_PUNTA = 10.0  # segundos

LIMITE_X_MIN = -0.390
LIMITE_X_MAX = 0.160
LIMITE_Y_MIN = -0.760 
LIMITE_Y_MAX = -0.265

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def generar_direcciones(n=15):
    direcciones = []
    for i in range(n):
        angulo = 2 * np.pi * i / n
        dx = np.cos(angulo)
        #dy = -np.sin(angulo)  # sentido horario
        dy = np.sin(angulo) # sentido antihorario
        direcciones.append((dx, dy))
    return direcciones

def ejecutar_salto_exploratorio(rtde_c, rtde_r, pose_actual, Z_PLANO, rotacion_abajo, LIMITE_X_MIN, LIMITE_X_MAX, LIMITE_Y_MIN, LIMITE_Y_MAX, DELAY_LECTURA):
    print("Ejecutando salto exploratorio de 15 cm...")
    angulo_salto = np.random.uniform(0, 2 * np.pi)
    dx_salto = np.cos(angulo_salto)
    dy_salto = np.sin(angulo_salto)

    x_salto = pose_actual[0] + dx_salto * 0.15
    y_salto = pose_actual[1] + dy_salto * 0.15

    x_salto = np.clip(x_salto, LIMITE_X_MIN, LIMITE_X_MAX)
    y_salto = np.clip(y_salto, LIMITE_Y_MIN, LIMITE_Y_MAX)

    pose_salto = [x_salto, y_salto, Z_PLANO] + rotacion_abajo
    rtde_c.moveL(pose_salto, speed=0.2, acceleration=0.2)
    time.sleep(DELAY_LECTURA)

    nueva_pose = rtde_r.getActualTCPPose()
    print(f"Salto ejecutado a: x={x_salto:.3f}, y={y_salto:.3f}")
    return nueva_pose

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# mirar hacia abajo
rotacion_abajo = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec().tolist()

puntos_mallado = cargar_mallado()
print(f"✅ Cargados {len(puntos_mallado)} puntos del mallado")

pos_inicial = [-0.11947692474323461, -0.26439650781719, -0.2773488558014402, 3.140198214586112, 0.0064768660533132535, 0.0062702868250280284]
memoria = {}

# puntos especiales
PUNTOS_ESPECIALES = {f"p{i}" for i in range(18, 24)}  # p18, p19, p20, p21, p22, p23

for target_key in sorted(puntos_mallado.keys(), key=lambda x: int(x[1:])):
    if target_key in memoria:
        print(f"\n{target_key} ya alcanzado. Saltando.")
        continue

    ESPECIAL = target_key in PUNTOS_ESPECIALES
    TOL_LOCAL = 22 if ESPECIAL else TOL_PIXEL

    print("Moviendo a posición inicial...")
    rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)
    pose_actual = rtde_r.getActualTCPPose()
    HISTORIAL = [pose_actual.copy()]

    target_2d = puntos_mallado[target_key]
    print(f"\n{'='*60}")
    print(f"INICIANDO exploración para: {target_key} = ({target_2d[0]:.1f}, {target_2d[1]:.1f})")
    print(f"{'='*60}")

    paso = 0
    paso_actual = PASO_INICIAL
    fallas_consecutivas = 0
    direccion_actual = None

    def buscar_punto_cercano_no_visitado(punta, umbral=12):
        for pk, pv in puntos_mallado.items():
            if pk in memoria:
                continue
            if distancia_2d(punta, pv) <= umbral:
                return pk, pv
        return None, None

    try:
        while paso < MAX_PASOS:
            tiempo_inicio_sin_punta = time.time()
            punta_actual = leer_punta_robot()
            
            # hsperar hasta 10 segundos si no hay punta
            while punta_actual is None:
                if time.time() - tiempo_inicio_sin_punta > TIEMPO_MAX_SIN_PUNTA:
                    print("10 segundos sin detectar punta roja. Saltando exploratoriamente.")
                    pose_actual = ejecutar_salto_exploratorio(
                        rtde_c, rtde_r, pose_actual, Z_PLANO, rotacion_abajo,
                        LIMITE_X_MIN, LIMITE_X_MAX, LIMITE_Y_MIN, LIMITE_Y_MAX, DELAY_LECTURA
                    )
                    HISTORIAL.append(pose_actual.copy())
                    direccion_actual = None
                    paso += 1
                    break
                else:
                    print("No se detecta la punta roja. Esperando...")
                    time.sleep(DELAY_LECTURA)
                    punta_actual = leer_punta_robot()

            if punta_actual is None:
                continue

            dist_actual = distancia_2d(punta_actual, target_2d)
            if dist_actual < TOL_LOCAL:
                print("¡Objetivo principal alcanzado!")
                memoria[target_key] = rtde_r.getActualTCPPose()
                rtde_c.moveL(memoria[target_key], speed=0.15, acceleration=0.15)
                print("Esperando 5 segundos en el objetivo...")
                time.sleep(5)
                break

            # siempre verificar puntos cercanos no visitados
            punto_cercano_key, _ = buscar_punto_cercano_no_visitado(punta_actual, umbral=12)
            if punto_cercano_key is not None and punto_cercano_key != target_key:
                print(f"\n¡Punto cercano: {punto_cercano_key}!")
                memoria[punto_cercano_key] = rtde_r.getActualTCPPose()
                rtde_c.moveL(memoria[punto_cercano_key], speed=0.15, acceleration=0.15)
                time.sleep(5)
                pose_actual = rtde_r.getActualTCPPose()
                HISTORIAL.append(pose_actual.copy())
                punta_check = leer_punta_robot()
                if punta_check is not None and distancia_2d(punta_check, target_2d) < TOL_LOCAL:
                    print("Objetivo alcanzado tras guardar punto.")
                    memoria[target_key] = pose_actual
                    rtde_c.moveL(memoria[target_key], speed=0.15, acceleration=0.15)
                    time.sleep(5)
                    break

            print(f"\n--- Paso {paso + 1} (paso={paso_actual:.3f} m) ---")
            print(f"Distancia: {dist_actual:.1f} px")

            # paso adaptativo para los puntos especiales para evitar problema de limites
            if ESPECIAL and dist_actual < 30:
                paso_usar = 0.03
            else:
                paso_usar = 0.06 if dist_actual < 25 else paso_actual

            if direccion_actual is not None:
                direcciones = [direccion_actual] + generar_direcciones(15)
            else:
                direcciones = generar_direcciones(15)

            mejora_encontrada = False

            for dx, dy in direcciones:
                x1 = pose_actual[0] + dx * paso_usar
                y1 = pose_actual[1] + dy * paso_usar

                if not (LIMITE_X_MIN <= x1 <= LIMITE_X_MAX): continue
                if not (LIMITE_Y_MIN <= y1 <= LIMITE_Y_MAX): continue

                pose1 = [x1, y1, Z_PLANO] + rotacion_abajo
                rtde_c.moveL(pose1, speed=0.15, acceleration=0.15)
                time.sleep(DELAY_LECTURA)

                punta1 = leer_punta_robot()
                if punta1 is None:
                    rtde_c.moveL([pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo, speed=0.15, acceleration=0.15)
                    time.sleep(0.1)
                    continue

                dist1 = distancia_2d(punta1, target_2d)

                if dist1 <= dist_actual - MEJORA_MINIMA:
                    print(f"¡Mejora! {dist_actual:.1f} → {dist1:.1f} px")
                    pose_actual = rtde_r.getActualTCPPose()
                    HISTORIAL.append(pose_actual.copy())
                    direccion_actual = (dx, dy)
                    mejora_encontrada = True
                    break
                else:
                    rtde_c.moveL([pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo, speed=0.15, acceleration=0.15)
                    time.sleep(0.1)

            if mejora_encontrada:
                fallas_consecutivas = 0
                paso += 1
            else:
                pose_actual = ejecutar_salto_exploratorio(
                    rtde_c, rtde_r, pose_actual, Z_PLANO, rotacion_abajo,
                    LIMITE_X_MIN, LIMITE_X_MAX, LIMITE_Y_MIN, LIMITE_Y_MAX, DELAY_LECTURA
                )
                HISTORIAL.append(pose_actual.copy())
                direccion_actual = None
                fallas_consecutivas = 0
                paso += 1

        else:
            print(f"\nLímite de pasos alcanzado para {target_key}.")

    except Exception as e:
        print(f"Error en {target_key}: {e}")

print("\nVolviendo a posición inicial final...")
rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)

try:
    rtde_c.disconnect()
    rtde_r.disconnect()
except:
    pass

try:
    with open(os.path.join(OUTPUT_DIR, "memoria_espacial.txt"), "w") as f: # guardar resultados
        for k in sorted(memoria.keys(), key=lambda x: int(x[1:])):
            pose = memoria[k]
            x, y, z, rx, ry, rz = pose
            f.write(f"{k} {x:.6f} {y:.6f} {z:.6f} {rx:.6f} {ry:.6f} {rz:.6f}\n")
    print(f"\nResultados guardados en: {os.path.join(OUTPUT_DIR, 'memoria_espacial.txt')}")
except Exception as e:
    print(f"\nError al guardar resultados: {e}")

print(f"\n{'='*60}")
print("RESULTADO FINAL")
print(f"{'='*60}")
if memoria:
    print(f"Puntos alcanzados: {len(memoria)}/{len(puntos_mallado)}")
    for k in sorted(memoria.keys(), key=lambda x: int(x[1:])):
        v = memoria[k]
        print(f"{k}: x={v[0]:.4f}, y={v[1]:.4f}")
else:
    print("Ningún punto alcanzado.")

print("\nPrograma terminado.")

# printear el tiempo
duracion_total = time.time() - inicio_ejecucion
horas = int(duracion_total // 3600)
minutos = int((duracion_total % 3600) // 60)
segundos = int(duracion_total % 60)
print(f"\nDuración de exploración: {horas:02d}:{minutos:02d}:{segundos:02d}")