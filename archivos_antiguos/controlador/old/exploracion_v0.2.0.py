import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

#ROBOT_IP = "192.168.19.128"  # simulador
ROBOT_IP = "169.254.129.110"  # robot real

# --- leer punta_robot.txt ---
punta_robot_2d = None
if os.path.exists("punta_robot.txt"):
    with open("punta_robot.txt", "r") as f:
        line = f.readline().strip()
        if line and line != "None":
            try:
                x_str, y_str = line.split()
                punta_robot_2d = [float(x_str), float(y_str)]
                print(f"Punto rojo detectado: x={punta_robot_2d[0]}, y={punta_robot_2d[1]}")
            except Exception as e:
                print(f"Error al leer punta_robot.txt: {e}")
        else:
            print("No se detecta la punta")
else:
    print("Archivo 'punta_robot.txt' no encontrado")

# --- leer puntos_mallado.txt y crear diccionario ---
puntos_mallado = {}
if os.path.exists("puntos_mallado.txt"):
    with open("puntos_mallado.txt", "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    x_str, y_str = line.split()
                    puntos_mallado[f"p{idx}"] = [float(x_str), float(y_str)]
                except Exception as e:
                    print(f"Línea {idx} inválida en puntos_mallado.txt: {e}")
    print(f"Cargados {len(puntos_mallado)} puntos del mallado")
else:
    print("Archivo 'puntos_mallado.txt' no encontrado")

if punta_robot_2d:
    print(f"\nPosición 2D de la punta roja: {punta_robot_2d}")

if puntos_mallado:
    print("Primeros 3 puntos del mallado:")
    for i in range(min(3, len(puntos_mallado))):
        print(f"  p{i} = {puntos_mallado[f'p{i}']}")
        
esquinas_cartesianas = [
    [0.161, -0.781, -0.277],   # esquina A
    [-0.383, -0.700, -0.277],  # esquina B
    [-0.394, -0.269, -0.277],  # esquina C
    [0.147, -0.361, -0.277],   # esquina D
]

# para hacer pi, 0, 0 (apuntar hacia abajo)
rotacion_abajo = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec().tolist()

pos_inicial_grados = [-87, -183, -130, 43, 90, 183]
pos_inicial = np.radians(pos_inicial_grados).tolist()

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

try:
    # ir a la posición inicial
    rtde_c.moveJ(pos_inicial, speed=0.5, acceleration=0.5)
    pos_inicialxyz = rtde_r.getActualTCPPose()
    print(f"Posición inicial TCP: x={pos_inicialxyz[0]:.3f}, y={pos_inicialxyz[1]:.3f}, z={pos_inicialxyz[2]:.3f}")

    for i, (x, y, z) in enumerate(esquinas_cartesianas):
        print(f"\nMoviendo a esquina {i+1}: x={x}, y={y}, z={z}")
        pose = [x, y, z] + rotacion_abajo
        rtde_c.moveL(pose, speed=0.2, acceleration=0.2)

    # volver a la posición inicial
    rtde_c.moveJ(pos_inicial, speed=0.5, acceleration=0.5)

finally:
    rtde_c.disconnect()
    rtde_r.disconnect()