import rtde_control, rtde_receive
import cv2, numpy as np
import time, csv, datetime, os, sys

# importamos las funciones y librerias necesarias
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from modulos_vision.seguimiento_mano import detectar_punta_verde, triangular

# configuracion de ip
ROBOT_IP = "169.254.129.110"
#ROBOT_IP = "192.168.253.131"

# conexion al cobot
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

pose_inicio = [-0.11557, -0.4964, -0.01976, 3.140198214586112, 0, 0]
rot = pose_inicio[3:]
ox, oy, oz = pose_inicio[:3]


print("Moviendo a posición de inicio...")
rtde_c.moveL(pose_inicio, 0.1, 0.1)
time.sleep(1)

# topes maximos fisicos del brazo (seguridad)
LIMITE_X_MIN = -0.390
LIMITE_X_MAX = 0.160
LIMITE_Y_MIN = -0.760 
LIMITE_Y_MAX = -0.265
LIMITE_Z_MIN = -0.200  
LIMITE_Z_MAX = 0.211   

# funcion para crear el cubo de puntos a recorrer
def crear_grid():
    
    xs = np.linspace(LIMITE_X_MIN + 0.05, LIMITE_X_MAX - 0.05, 3) 
    ys = np.linspace(LIMITE_Y_MIN + 0.05, LIMITE_Y_MAX - 0.05, 3)
    zs = np.linspace(LIMITE_Z_MIN + 0.02, LIMITE_Z_MAX - 0.02, 3) 

    
    return [[xi, yi, zi] for zi in zs for yi in ys for xi in xs]

grid = crear_grid()

# abrimos las camaras web
cam_i = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam_d = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# preparamos el archivo csv donde guardaremos el diccionario de mapeo
output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)
ruta = os.path.join(output_dir, f"mapeo_{datetime.datetime.now().strftime('%H%M%S')}.csv")

with open(ruta, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')  
    writer.writerow(["ID","Robot_X","Robot_Y","Robot_Z","Cam_X","Cam_Y","Cam_Z"])

    for i, p in enumerate(grid):
        print(f"[{i+1}/27] -> {p}")
        try:
            rtde_c.moveL(p + rot, 0.1, 0.1)
        except:
            continue

        time.sleep(1)
        pose = rtde_r.getActualTCPPose()

       #Reintentar durante unos segundos si no se ve la punta
        vision_ok = False
        tiempo_maximo_espera = 10  
        tiempo_inicio = time.time()
        
        while time.time() - tiempo_inicio < tiempo_maximo_espera:
            ret_i, fi = cam_i.read()
            ret_d, fd = cam_d.read()
            
            if ret_i and ret_d:
                fi, fd = cv2.flip(fi,1), cv2.flip(fd,1)
                pi = detectar_punta_verde(fi)
                pd = detectar_punta_verde(fd)
                coord, _ = triangular(pi, pd, 500, None)
                
                if coord:
                    cx, cy, cz = coord
                    vision_ok = True
                    break 
            
            time.sleep(0.5)
            
        if not vision_ok:
            print(f"Punta no detectada en la posición {i}. Descartando captura.")
            writer.writerow([i, pose[0], pose[1], pose[2], "ERROR", "ERROR", "ERROR"])
            continue 

        writer.writerow([i, pose[0], pose[1], pose[2], cx, cy, cz])

rtde_c.moveL(pose_inicio, 0.1, 0.1)
cam_i.release()
cam_d.release()
rtde_c.disconnect()
rtde_r.disconnect()
print(f"MAPEO TERMINADO → {ruta}")