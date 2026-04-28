import time
import xmlrpc.client

# IP del robot
ROBOT_IP = "169.254.129.110"

print("--- TEST DE PINZA ONROBOT (ORDEN: ID, W, F) ---")

try:
    print(f"Conectando al servidor XML-RPC en {ROBOT_IP}:41414...")
    gripper = xmlrpc.client.ServerProxy(f"http://{ROBOT_IP}:41414/")
    
    # Verificamos conexión
    width = gripper.rg_get_width(0)
    print(f"¡Conexión exitosa! Anchura actual: {width} mm")

    print("\nIntentando con orden (ID: Int, Anchura: Float, Fuerza: Float)...")
    # rg_grip(id_int, anchura_float, fuerza_float)
    gripper.rg_grip(0, 0.0, 40.0)
    time.sleep(4)

    print("\nIntentando abrir pinza...")
    gripper.rg_grip(0, 110.0, 40.0)
    time.sleep(4)

except Exception as e:
    print(f"\n[ERROR] Fallo en la comunicación: {e}")

print("\n--- TEST FINALIZADO ---")
