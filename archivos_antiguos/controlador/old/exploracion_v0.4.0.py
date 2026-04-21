import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

# ROBOT_IP = "192.168.19.128"  # simulador
ROBOT_IP = "169.254.129.110"  # robot real

# --- Parámetros ---
Z_PLANO = -0.277
PASO_INICIAL = 0.06      # 5 cm
PASO_MIN = 0.01          # 1 cm
MAX_FALLAS = 3
MAX_PASOS = 50
TOL_PIXEL = 18           # reducido para mayor precisión
DELAY_LECTURA = 0.6

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

# --- Posición inicial (pose cartesiana completa) ---
pos_inicial = [-0.11947692474323461, -0.26439650781719, -0.2773488558014402, 3.140198214586112, 0.0064768660533132535, 0.0062702868250280284]
memoria = {}

# --- Recorrer todos los puntos del mallado en orden ---
for target_key in sorted(puntos_mallado.keys(), key=lambda x: int(x[1:])):  # p0, p1, p2, ...
    if target_key in memoria:
        print(f"\n⏭️  {target_key} ya alcanzado. Saltando.")
        continue

    # ✅ SIEMPRE volver a posición inicial antes de explorar un nuevo objetivo
    print("🏠 Moviendo a posición inicial antes de explorar...")
    rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)
    pose_actual = rtde_r.getActualTCPPose()
    HISTORIAL = [pose_actual.copy()]

    target_2d = puntos_mallado[target_key]
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO exploración para: {target_key} = ({target_2d[0]:.1f}, {target_2d[1]:.1f})")
    print(f"{'='*60}")

    paso = 0
    paso_actual = PASO_INICIAL
    fallas_consecutivas = 0
    direccion_actual = None

    # Función interna para buscar punto no visitado cercano
    def buscar_punto_cercano_no_visitado(punta, umbral=12):
        for pk, pv in puntos_mallado.items():
            if pk in memoria:
                continue
            if distancia_2d(punta, pv) <= umbral:
                return pk, pv
        return None, None

    try:
        while paso < MAX_PASOS:
            punta_actual = leer_punta_robot()
            if punta_actual is None:
                print("⚠️ No se detecta la punta roja. Esperando...")
                time.sleep(DELAY_LECTURA)
                continue

            # 🔍 Verificar si hay algún punto NO visitado muy cercano (≤10 px)
            punto_cercano_key, punto_cercano_2d = buscar_punto_cercano_no_visitado(punta_actual, umbral=12)
            if punto_cercano_key is not None and punto_cercano_key != target_key:
                print(f"\n✨ ¡Detectado punto cercano no visitado: {punto_cercano_key}!")
                dist_cercana = distancia_2d(punta_actual, punto_cercano_2d)
                print(f"   Distancia: {dist_cercana:.1f} px")
                print("   Guardando y esperando 5 segundos...")

                memoria[punto_cercano_key] = rtde_r.getActualTCPPose()
                rtde_c.moveL(memoria[punto_cercano_key], speed=0.15, acceleration=0.15)
                time.sleep(5)

                pose_actual = rtde_r.getActualTCPPose()
                HISTORIAL.append(pose_actual.copy())
                print(f"✅ {punto_cercano_key} registrado. Continuando hacia {target_key}...")

            # Evaluar distancia al objetivo principal
            dist_actual = distancia_2d(punta_actual, target_2d)
            print(f"\n--- Paso {paso + 1} (paso_base={paso_actual:.3f} m) ---")
            print(f"📍 Distancia actual a {target_key}: {dist_actual:.1f} px")

            paso_usar = 0.03 if dist_actual < 25 else paso_actual
            print(f"    ➡️  Paso dinámico aplicado: {paso_usar:.3f} m")

            if dist_actual < TOL_PIXEL:
                print("✅ ¡Objetivo principal alcanzado!")
                memoria[target_key] = rtde_r.getActualTCPPose()
                rtde_c.moveL(memoria[target_key], speed=0.15, acceleration=0.15)
                print("⏸️  Esperando 5 segundos en el objetivo...")
                time.sleep(5)
                break

            # --- Exploración en 15 direcciones ---
            if direccion_actual is not None:
                direcciones = [direccion_actual] + generar_direcciones(15)
            else:
                direcciones = generar_direcciones(15)

            mejora_confirmada = False
            mejor_pose1 = None
            direccion_confirmada = None

            for dx, dy in direcciones:
                x1 = pose_actual[0] + dx * paso_usar
                y1 = pose_actual[1] + dy * paso_usar

                if not (LIMITE_X_MIN <= x1 <= LIMITE_X_MAX): continue
                if not (LIMITE_Y_MIN <= y1 <= LIMITE_Y_MAX): continue

                pose1 = [x1, y1, Z_PLANO] + rotacion_abajo
                rtde_c.moveL(pose1, speed=0.15, acceleration=0.15)
                time.sleep(DELAY_LECTURA)
                punta1 = leer_punta_robot()
                
                # 🔍 NUEVO: verificar puntos cercanos en pose1
                if punta1 is not None:
                    pc_key, pc_2d = buscar_punto_cercano_no_visitado(punta1, umbral=12)
                    if pc_key is not None and pc_key != target_key:
                        print(f"\n✨ ¡Punto cercano detectado en pose1: {pc_key}!")
                        memoria[pc_key] = rtde_r.getActualTCPPose()
                        rtde_c.moveL(memoria[pc_key], speed=0.15, acceleration=0.15)
                        time.sleep(5)
                        pose_actual = rtde_r.getActualTCPPose()
                        HISTORIAL.append(pose_actual.copy())
                        print(f"✅ {pc_key} registrado. Reiniciando exploración desde aquí.")
                        mejora_confirmada = False  # cancelar esta exploración
                        break  # salir del for de direcciones

                dist1 = distancia_2d(punta1, target_2d) if punta1 else float('inf')

                if dist1 >= dist_actual:
                    rtde_c.moveL([pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo, speed=0.15, acceleration=0.15)
                    time.sleep(0.1)
                    continue

                x2 = x1 + dx * paso_usar
                y2 = y1 + dy * paso_usar

                if not (LIMITE_X_MIN <= x2 <= LIMITE_X_MAX) or not (LIMITE_Y_MIN <= y2 <= LIMITE_Y_MAX):
                    if dist1 < dist_actual - 5.0:
                        print(f"🟡 Mejora en borde: aceptando (mejora ≥5 px)")
                        mejor_pose1 = pose1.copy()
                        direccion_confirmada = (dx, dy)
                        rtde_c.moveL([pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo, speed=0.15, acceleration=0.15)
                        time.sleep(0.1)
                        mejora_confirmada = True
                        break
                    else:
                        rtde_c.moveL([pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo, speed=0.15, acceleration=0.15)
                        time.sleep(0.1)
                        continue

                pose2 = [x2, y2, Z_PLANO] + rotacion_abajo
                rtde_c.moveL(pose2, speed=0.15, acceleration=0.15)
                time.sleep(DELAY_LECTURA)
                punta2 = leer_punta_robot()

                # 🔍 NUEVO: verificar puntos cercanos en pose2
                if punta2 is not None:
                    pc_key, pc_2d = buscar_punto_cercano_no_visitado(punta2, umbral=12)
                    if pc_key is not None and pc_key != target_key:
                        print(f"\n✨ ¡Punto cercano detectado en pose2: {pc_key}!")
                        memoria[pc_key] = rtde_r.getActualTCPPose()
                        rtde_c.moveL(memoria[pc_key], speed=0.15, acceleration=0.15)
                        time.sleep(5)
                        pose_actual = rtde_r.getActualTCPPose()
                        HISTORIAL.append(pose_actual.copy())
                        print(f"✅ {pc_key} registrado. Reiniciando exploración desde aquí.")
                        mejora_confirmada = False
                        break

                dist2 = distancia_2d(punta2, target_2d) if punta2 else float('inf')

                rtde_c.moveL([pose_actual[0], pose_actual[1], Z_PLANO] + rotacion_abajo, speed=0.15, acceleration=0.15)
                time.sleep(0.1)

                if dist1 < dist_actual - 5.0:
                    mejora_px = dist_actual - dist1
                    print(f"🟢 ¡Mejora significativa! {mejora_px:.1f} px ({dist_actual:.1f} → {dist1:.1f})")
                    mejor_pose1 = pose1.copy()
                    direccion_confirmada = (dx, dy)
                    mejora_confirmada = True
                    break

            if mejora_confirmada:
                rtde_c.moveL(mejor_pose1, speed=0.15, acceleration=0.15)
                pose_actual = rtde_r.getActualTCPPose()
                HISTORIAL.append(pose_actual.copy())
                direccion_actual = direccion_confirmada
                fallas_consecutivas = 0
                paso += 1
            else:
                fallas_consecutivas += 1
                print(f"🔴 Sin mejora confirmada. Fallas consecutivas: {fallas_consecutivas}")

                if fallas_consecutivas >= MAX_FALLAS:
                    paso_actual = max(paso_actual * 0.5, PASO_MIN)
                    fallas_consecutivas = 0
                    print(f"🔄 Paso base reducido a {paso_actual:.3f} m")
                    direccion_actual = None
                else:
                    if len(HISTORIAL) > 1:
                        HISTORIAL.pop()
                        pose_anterior = HISTORIAL[-1]
                        print("⏪ Retrocediendo al último punto válido...")
                        rtde_c.moveL(pose_anterior, speed=0.15, acceleration=0.15)
                        pose_actual = rtde_r.getActualTCPPose()
                        direccion_actual = None

        else:
            print(f"\n⚠️ Límite de {MAX_PASOS} pasos alcanzado para {target_key}. No se guardará.")

    except Exception as e:
        print(f"❌ Error durante la exploración de {target_key}: {e}")

# --- Finalizar ---
print("\n🏠 Volviendo a posición inicial final...")
rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)

try:
    rtde_c.disconnect()
    rtde_r.disconnect()
except:
    pass

# --- Resultado final ---
print(f"\n{'='*60}")
print("📊 RESULTADO FINAL")
print(f"{'='*60}")
if memoria:
    print(f"✅ Puntos alcanzados: {len(memoria)}/{len(puntos_mallado)}")
    print("\n=== MEMORIA COMPLETA (x, y, z, rx, ry, rz) ===")
    for k in sorted(memoria.keys(), key=lambda x: int(x[1:])):
        v = memoria[k]
        print(f"{k}: x={v[0]:.6f}, y={v[1]:.6f}, z={v[2]:.6f}, rx={v[3]:.6f}, ry={v[4]:.6f}, rz={v[5]:.6f}")
else:
    print("❌ Ningún punto fue alcanzado.")

print("\n✅ Programa terminado.")