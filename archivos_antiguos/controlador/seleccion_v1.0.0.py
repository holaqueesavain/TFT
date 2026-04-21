import os
import time
import rtde_control
import rtde_receive

#ROBOT_IP = "169.254.129.110"  # robot real
ROBOT_IP = "192.168.19.128"  # simulador

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OBJETOS_PATH = "C:\\ARIS\\universidad\\TFG\\Programa\\sistemavision\\output\\objetos.txt"

pos_inicial = [-0.11947692474323461, -0.26439650781719, -0.2773488558014402,
               3.140198214586112, 0.0064768660533132535, 0.0062702868250280284]

def cargar_memoria_espacial():
    """Carga memoria_espacial.txt en un diccionario."""
    memoria = {}
    ruta = os.path.join(OUTPUT_DIR, "memoria_espacial.txt")
    if not os.path.exists(ruta):
        print(f"Archivo no encontrado: {ruta}")
        return memoria

    with open(ruta, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if len(partes) < 7:
                print(f"Línea inválida ignorada: {linea}")
                continue
            clave = partes[0]
            try:
                valores = [float(x) for x in partes[1:7]]
                memoria[clave] = valores
            except ValueError:
                print(f"Error al convertir valores en: {linea}")
                continue
    return memoria

def leer_contenido_archivo(ruta):
    """Devuelve el contenido completo del archivo como string, o None si no existe."""
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r") as f:
        return f.read()

def extraer_etiquetas_y_alturas(contenido):
    """
    Extrae tripletas (etiqueta_punto, altura, tipo_objeto) de cada línea.
    """
    if contenido is None:
        return []
    tripletas = []
    for linea in contenido.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        partes = linea.split()
        etiqueta_punto = partes[0]
        altura = 0.20  # valor por defecto (para evitar colision en caso de no especificar altura)
        tipo_objeto = "desconocido"

        if len(partes) >= 2:
            try:
                altura = float(partes[1])
            except ValueError:
                print(f"Altura inválida en línea: {linea}. Usando 0.20 m.")
        if len(partes) >= 3:
            tipo_objeto = partes[2]  # se guarda, pero no se usa

        tripletas.append((etiqueta_punto, altura, tipo_objeto))
    return tripletas

def desconectar_robot():
    """Desconecta interfaces del robot de forma segura."""
    try:
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("\nInterfaces del robot desconectadas.")
    except:
        pass

def main():
    print("Cargando memoria espacial...")
    memoria = cargar_memoria_espacial()
    if not memoria:
        print("No se pudo cargar la memoria espacial. Abortando.")
        desconectar_robot()
        return

    print("Iniciando secuencia de robot...")
    print("Presiona Ctrl+C en la terminal para detener el programa.")

    print("Moviendo a pose inicial...")
    rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)

    contenido_anterior = None

    try:
        while True:
            contenido_actual = leer_contenido_archivo(OBJETOS_PATH)

            if contenido_actual != contenido_anterior:
                contenido_anterior = contenido_actual
                objetivos = extraer_etiquetas_y_alturas(contenido_actual)

                if not objetivos:
                    print("objetos.txt vacío. Esperando nuevas detecciones...")
                else:
                    print(f"Nuevos objetos detectados: {objetivos}")
                    for etiqueta, altura_obj, tipo_objeto in objetivos:
                        if etiqueta in memoria:
                            pose_base = memoria[etiqueta][:]  # copia
                            pose_ajustada = pose_base.copy()
                            pose_ajustada[2] += altura_obj  # sumar altura a Z
                            
                            print(f"Procesando {etiqueta} → altura={altura_obj:.4f} m → pose ajustada: {pose_ajustada}")
                            rtde_c.moveL(pose_ajustada, speed=0.3, acceleration=0.3)
                            print(f"Esperando 3 segundos sobre {etiqueta}...")
                            time.sleep(3)
                            print("Regresando a pose inicial...")
                            rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)
                        else:
                            print(f"Etiqueta '{etiqueta}' no encontrada en memoria. Saltando.")
                    print("Secuencia completada. Esperando cambios en objetos.txt...")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSecuencia interrumpida por el usuario (Ctrl+C).")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        print("Volviendo a pose inicial...")
        try:
            rtde_c.moveL(pos_inicial, speed=0.3, acceleration=0.3)
        except:
            pass
        desconectar_robot()
        print("Programa finalizado.")

if __name__ == "__main__":
    main()