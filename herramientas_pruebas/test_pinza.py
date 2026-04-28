"""
Prueba de comunicación con la pinza OnRobot RG2.
Se verifica el correcto funcionamiento de los comandos de apertura y cierre.
"""

import time
import xmlrpc.client

ROBOT_IP = "169.254.129.110"

try:
    print(f"Conectando con el servidor del robot ({ROBOT_IP}:41414)...")
    gripper = xmlrpc.client.ServerProxy(f"http://{ROBOT_IP}:41414/")
    
    width = gripper.rg_get_width(0)
    print(f"Conexión OK. Anchura actual de la pinza: {width} mm")

    print("\nProbando cierre")
    gripper.rg_grip(0, 0.0, 40.0)
    time.sleep(4)

    print("\nProbando apertura")
    gripper.rg_grip(0, 110.0, 40.0)
    time.sleep(4)

except Exception as e:
    print(f"\nError en el test: {e}")


