import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

#ROBOT_IP = "192.168.19.128"  # simulador
ROBOT_IP = "169.254.129.110"  # robot real

# --- Parámetros ---
Z_PLANO = -0.277
PASO = 0.01
MAX_INTENTOS = 50
TOL_PIXEL = 15
DELAY_LECTURA = 0.3

# --- Límites del espacio de trabajo (en metros) ---
LIMITE_X_MIN = -0.390
LIMITE_X_MAX = 0.160
LIMITE_Y_MIN = -0.815 
LIMITE_Y_MAX = -0.265

# --- Leer archivo de mallado ---
def cargar_mallado(filepath="C:\\ARIS\\universidad\\TFG\\Programa\\prototipovision_v0.1\\puntos_mallado.txt"):
    puntos = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo {filepath} no encontrado")
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                x, y = map(float, line.strip().split())
                puntos[f"p{i}"] = [x, y]
    return puntos

# --- Leer posición actual de la punta roja ---
def leer_punta_robot(filepath="C:\\ARIS\\universidad\\TFG\\Programa\\prototipovision_v0.1\\punta_robot.txt"):
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

# --- Calcular distancia en píxeles ---
def distancia_2d(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- Generar 15 direcciones en el plano XY ---
def generar_direcciones(n=15):
    direcciones = []
    for i in range(n):
        angulo = 2 * np.pi * i / n
        dx = np.cos(angulo)
        dy = np.sin(angulo)
        direcciones.append((dx, dy))
    return direcciones

# --- Conexiones ---
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# --- Rotación fija (mirando abajo) ---
rotacion_abajo = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec().tolist()

# --- Cargar mallado ---
puntos_mallado = cargar_mallado()
print(f"✅ Cargados {len(puntos_mallado)} puntos del mallado")

# --- Elegir objetivo ---
target_key = "p3"
if target_key not in puntos_mallado:
    raise ValueError(f"Objetivo {target_key} no existe en el mallado")
target_2d = puntos_mallado[target_key]
print(f"🎯 Objetivo: {target_key} = {target_2d}")

# --- Ir a posición inicial segura ---
pos_inicial = [-0.11947692474323461, -0.26439650781719, -0.2773488558014402, 3.140198214586112, 0.0064768660533132535, 0.0062702868250280284]
rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)

# --- Obtener pose inicial del TCP ---
pose_actual = rtde_r.getActualTCPPose()
x0, y0 = pose_actual[0], pose_actual[1]
print("Pose inicial:", pose_actual)

memoria = {}
intentos = 0
mejora = True

while intentos < MAX_INTENTOS and mejora:
    print(f"\n--- Intento {intentos + 1} ---")
    
    punta_actual = leer_punta_robot()
    if punta_actual is None:
        print("⚠️ No se detecta la punta roja. Esperando...")
        time.sleep(DELAY_LECTURA)
        continue

    dist_actual = distancia_2d(punta_actual, target_2d)
    print(f"📍 Distancia actual a {target_key}: {dist_actual:.1f} px")

    if dist_actual < TOL_PIXEL:
        print("✅ ¡Objetivo alcanzado!")
        # Guardar la pose actual en el diccionario
        memoria[target_key] = rtde_r.getActualTCPPose()
        print(f"💾 Guardado en memoria: {target_key}")

        # 🔄 Volver EXACTAMENTE a la pose guardada EN EL DICCIONARIO
        print("➡️  Volviendo a la pose exacta del objetivo (desde memoria)...")
        rtde_c.moveL(memoria[target_key], speed=0.1, acceleration=0.1)

        # ⏳ Esperar 5 segundos
        print("⏸️  Esperando 5 segundos en el objetivo...")
        time.sleep(5)

        # 🏠 Volver a la pose inicial
        print("🏠 Volviendo a la posición inicial...")
        rtde_c.moveL(pos_inicial, speed=0.5, acceleration=0.5)

        break

    direcciones = generar_direcciones(15)
    mejor_direccion = None
    mejor_distancia = dist_actual
    pose_mejor = None

    for dx, dy in direcciones:
        x_new = pose_actual[0] + dx * PASO
        y_new = pose_actual[1] + dy * PASO

        # 🔒 VERIFICACIÓN DE LÍMITES DE SEGURIDAD
        if not (LIMITE_X_MIN <= x_new <= LIMITE_X_MAX):
            continue
        if not (LIMITE_Y_MIN <= y_new <= LIMITE_Y_MAX):
            continue

        pose_prueba = [x_new, y_new, Z_PLANO] + rotacion_abajo
        rtde_c.moveL(pose_prueba, speed=0.1, acceleration=0.1)

        time.sleep(DELAY_LECTURA)
        punta_nueva = leer_punta_robot()
        dist_nueva = distancia_2d(punta_nueva, target_2d) if punta_nueva else float('inf')

        if punta_nueva:
            print(f"  → Probando ({dx:+.2f}, {dy:+.2f}): punta=({punta_nueva[0]:.1f}, {punta_nueva[1]:.1f}), dist={dist_nueva:.1f} px")
        else:
            print(f"  → Probando ({dx:+.2f}, {dy:+.2f}): punta=NO DETECTADA, dist=∞")

        if dist_nueva < mejor_distancia:
            mejor_distancia = dist_nueva
            mejor_direccion = (dx, dy)
            pose_mejor = pose_prueba.copy()

        # Volver al punto actual
        pose_volver = [pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo
        rtde_c.moveL(pose_volver, speed=0.1, acceleration=0.1)
        time.sleep(0.2)

    if mejor_direccion is not None:
        print(f"✨ Mejora encontrada: distancia {dist_actual:.1f} → {mejor_distancia:.1f}")
        rtde_c.moveL(pose_mejor, speed=0.1, acceleration=0.1)
        pose_actual = rtde_r.getActualTCPPose()
        intentos += 1
    else:
        print("❌ No se encontró mejora en ninguna dirección (¿cerca de los límites?)")
        mejora = False

# --- Finalizar ---
rtde_c.moveL(pos_inicial, speed=0.5, acceleration=0.5)
rtde_c.disconnect()
rtde_r.disconnect()

if target_key in memoria:
    print(f"\n🎉 Éxito: {target_key} alcanzado y guardado.")
else:
    print(f"\n⚠️ Fracaso: no se alcanzó {target_key} tras {intentos} intentos.")

print("\n=== MEMORIA ===")
for k, v in memoria.items():
    print(f"{k}: [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f}, {v[4]:.6f}, {v[5]:.6f}]")