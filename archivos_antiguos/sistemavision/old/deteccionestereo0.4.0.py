import cv2
import numpy as np
import math
import time

# ----------------------------------------------------
# --------------------- V.0.4.0 ----------------------
# ----------------------------------------------------
"""Esta version incluye salidas de txt para mallado y 
deteccion de la punta del robot"""

points = []          
mesh_points = []  
obj_center = None    
H = None            
H_inv = None        
rectified_width, rectified_height = 350, 200  

last_update_time = 0

img_i = None      
img_f = None       
captured_i = False

capL = None
capR = None
stereo = None
Q = None  
disp_map = None 

def load_calibration():
    """Cargar calibración estéreo usando solo baseline y parámetros conocidos"""
    global Q, stereo, capL, capR

    baseline = 0.093       # Distancia entre cámaras en METROS
    focal_length_px = 900  # Longitud focal en PÍXELES
    img_width, img_height = 640, 480
    cx = img_width / 2     # Centro óptico X
    cy = img_height / 2    # Centro óptico Y

    """Esta matriz es una matriz de reproyeccion a 3D mediante dos cámaras estéreo que dan imagen en 2D."""
    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0,  focal_length_px],
        [0, 0, -1.0 / baseline, 0] 
    ])

    print(f"Matriz Q construida con baseline = {baseline} m, f = {focal_length_px} px")

    capL = cv2.VideoCapture(2)
    capR = cv2.VideoCapture(1)
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
        #cap.set(cv2.CAP_PROP_FOCUS, 0)  # enfoque manual

    if not capL.isOpened() or not capR.isOpened():
        print("Cámaras no disponibles")
        return False

    stereo = cv2.StereoSGBM_create(
        minDisparity=17, #old 0
        numDisparities=112, # old 128 --> 80
        blockSize=7, #old 7
        P1=1000, # old 8 * 3 * 7**2
        P2=4000, # old 32 * 3 * 7**2
        disp12MaxDiff=1, # old 1
        uniquenessRatio=0, #old 5
        speckleWindowSize=60, # old 0
        speckleRange=12 # old 0
    ) #parametros ajustados con calibracion_estereo.py
    return True

def get_disparity():
    """Capturar imágenes y calcular disparidad"""
    global disp_map
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        return None

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    try:
        disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        disp[disp <= 0] = 0.1
        disp_filtered = cv2.medianBlur(disp.astype(np.uint8), 5).astype(np.float32)
        disp_filtered[disp_filtered <= 0] = 0.1
        disp_filtered = cv2.GaussianBlur(disp_filtered, (5, 5), 0)
        disp_map = disp_filtered
        return disp_filtered
    except Exception as e:
        print(f"Error en disparidad: {e}")
        return None

def point_to_3d(x, y):
    """Convertir punto 2D (imagen izquierda) a 3D usando disparidad"""
    if disp_map is None:
        return None
    d = disp_map[int(y), int(x)]
    if d <= 0:
        return None
    point_3d = cv2.reprojectImageTo3D(disp_map, Q)[int(y), int(x)]
    point_3d[2] = -point_3d[2] 
    return point_3d 

def generate_triangle_points_in_rect(spacing=60):
    points_rect = []
    h = int(spacing * np.sqrt(3) / 2)
    for row in range(0, rectified_height + 1, h):
        offset = 0 if (row // h) % 2 == 0 else spacing // 2
        for col in range(offset, rectified_width + 1, spacing):
            points_rect.append((col, row))
    return points_rect

def point_in_triangle(pt, tri):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    b1 = sign(pt, tri[0], tri[1]) < 0.0
    b2 = sign(pt, tri[1], tri[2]) < 0.0
    b3 = sign(pt, tri[2], tri[0]) < 0.0
    return ((b1 == b2) and (b2 == b3))

def select_points(event, x, y, flags, param):
    global points, mesh_points, H, H_inv
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Punto {len(points)}: ({x}, {y})")
        if len(points) == 4:
            pts = np.array(points, dtype=np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            rect = np.zeros((4, 2), dtype=np.float32)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            dst = np.array([
                [0, 0],
                [rectified_width, 0],
                [rectified_width, rectified_height],
                [0, rectified_height]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(dst, rect)
            H_inv = cv2.getPerspectiveTransform(rect, dst)

            points_rect = generate_triangle_points_in_rect(spacing=60)
            points_rect_np = np.array(points_rect, dtype=np.float32).reshape(-1, 1, 2)
            points_img = cv2.perspectiveTransform(points_rect_np, H)
            mesh_points.clear()
            for pt in points_img:
                mesh_points.append((int(pt[0][0]), int(pt[0][1])))
            print(f"Malla generada con {len(mesh_points)} puntos")

            # escribir puntos mallado
            with open("puntos_mallado.txt", "w") as f:
                for pt in mesh_points:
                    f.write(f"{pt[0]} {pt[1]}\n")
            print("Archivo 'puntos_mallado.txt' actualizado.")

def show_depth_heatmap():
    if disp_map is None:
        print("No hay disparidad calculada")
        return
    depth_normalized = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    if len(mesh_points) > 0:
        for pt in mesh_points:
            cv2.circle(heatmap, pt, 3, (255, 255, 255), -1)
    cv2.imshow("Profundidad con Mesh", heatmap)
    cv2.waitKey(0)

if not load_calibration():
    exit()

cv2.namedWindow("Triangular Workspace")
cv2.setMouseCallback("Triangular Workspace", select_points)

print("Iniciando bucle principal. Presiona 'q' para salir.")

while True:
    ret, frame = capL.read()
    if not ret:
        break

    disp = get_disparity()

    # dibujar puntos seleccionados
    #for i, pt in enumerate(points):
    #    cv2.circle(frame, tuple(pt), 5, (0, 255, 255), -1)
    #    cv2.putText(frame, f"P{i+1}", tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for i, pt in enumerate(mesh_points):
        cv2.circle(frame, pt, 3, (255, 0, 0), -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    s_min, v_min = 29, 137
    s_max, v_max = 255, 255
    h_max_low = 6      # rango superior del primer segmento (0–6)
    h_min_high = 174   # rango inferior del segundo segmento (174–180)

    lower1 = np.array([0,       s_min, v_min])
    upper1 = np.array([h_max_low, s_max, v_max])

    lower2 = np.array([h_min_high, s_min, v_min])
    upper2 = np.array([180,     s_max, v_max])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)   
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_center = None
    min_area =27
    ar_min, ar_max = 0.24, 2.00

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h != 0 else 0

            if not (ar_min <= aspect_ratio <= ar_max):
                continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                obj_center = (cx, cy)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Punta robot", (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                break 
            
    current_time = time.time()
    if current_time - last_update_time >= 0.2:  # 200 ms
        with open("punta_robot.txt", "w") as f:
            if obj_center:
                f.write(f"{obj_center[0]} {obj_center[1]}\n")
            else:
                f.write("None\n")
        last_update_time = current_time

    if obj_center and len(mesh_points) > 0:
        min_dist = float('inf')
        closest_idx = -1
        for i, pt in enumerate(mesh_points):
            dist = np.linalg.norm(np.array(obj_center) - np.array(pt))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx != -1:
            cv2.circle(frame, mesh_points[closest_idx], 8, (0, 255, 255), 2)
            cv2.putText(frame, f"Robot cerca de: P{closest_idx}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(points) < 4:
        cv2.putText(frame, "Haz clic en 4 puntos para definir el area de trabajo", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'r' para reiniciar", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'v' para ver mesh 3D", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status = "Listo para 'i'" if not captured_i else "Listo para 'f'"
    cv2.putText(frame, f"Estado: {status}", (10, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    current_time = time.time()
    if current_time - last_update_time >= 0.2:  # 200 ms
        with open("punta_robot.txt", "w") as f:
            if obj_center:
                f.write(f"{obj_center[0]} {obj_center[1]}\n")
            else:
                f.write("None\n")
        last_update_time = current_time

    cv2.imshow("Triangular Workspace", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        points = []
        mesh_points = []
        H = None
        H_inv = None
        captured_i = False
        img_i = None
        img_f = None
        with open("puntos_mallado.txt", "w") as f:
            pass  # crea vacío o trunca
        print("\nSelección reiniciada y puntos_mallado.txt vaciado")
    elif key == ord('i'):
        retL, img_i = capL.read()
        if retL:
            captured_i = True
            print("\nImagen 'i' capturada")
        else:
            print("\nError al capturar imagen con 'i'")
    elif key == ord('f'):
        if not captured_i:
            print("\nPrimero presiona 'i' para tomar la primera imagen")
        else:
            retL, img_f = capL.read()
            if retL:
                print("\nImagen 'f' capturada. Calculando diferencia y profundidad...")

                gray_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
                gray_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray_i, gray_f)
                umbral_dif = 25
                mask_diff_original = np.uint8(diff > umbral_dif) * 255
                kernel = np.ones((5, 5), np.uint8)
                mask_diff_processed = cv2.morphologyEx(mask_diff_original, cv2.MORPH_CLOSE, kernel, iterations=1)
                contours, _ = cv2.findContours(mask_diff_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if disp_map is None:
                    print("No hay mapa de disparidad disponible")
                elif disp_map.shape != mask_diff_original.shape:
                    print("Disparidad y máscara tienen tamaños distintos")
                else:
                    disp_masked = np.where(mask_diff_original > 0, disp_map, 0)
                    contours, _ = cv2.findContours(mask_diff_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) == 0:
                        print("No se detectaron cambios significativos")
                        captured_i = False
                        img_i = None
                        img_f = None
                        continue

                    disp_vis = cv2.normalize(disp_masked, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

                    for i, pt in enumerate(mesh_points):
                        cv2.circle(heatmap, pt, 2, (255, 255, 255), -1)

                    obj_counter = 1
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = cv2.contourArea(cnt)
                        if area < 100:
                            continue
                        aspect_ratio = float(w) / h
                        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                            continue

                        cv2.rectangle(heatmap, (x, y), (x + w, y + h), (0, 255, 255), 1)
                        cv2.putText(heatmap, f"Obj {obj_counter}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        cx_bottom = x + w // 2
                        cy_bottom = y + h

                        xyz_bottom = point_to_3d(cx_bottom, cy_bottom)
                        if xyz_bottom is not None:
                            X, Y, Z = xyz_bottom
                            cv2.putText(heatmap, f"Z={Z:.2f}m", (x, y + h + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        if len(mesh_points) > 0:
                            min_dist = float('inf')
                            closest_mesh_idx = -1
                            for j, mesh_pt in enumerate(mesh_points):
                                dist = np.linalg.norm(np.array([cx_bottom, cy_bottom]) - np.array(mesh_pt))
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_mesh_idx = j

                            if closest_mesh_idx != -1:
                                closest_pt = mesh_points[closest_mesh_idx]
                                cv2.circle(heatmap, closest_pt, 6, (255, 255, 0), -1)
                                cv2.putText(heatmap, f"M{closest_mesh_idx}", (closest_pt[0] + 5, closest_pt[1] + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                                xyz_mesh = point_to_3d(closest_pt[0], closest_pt[1])
                                if xyz_mesh is not None:
                                    X_m, Y_m, Z_m = xyz_mesh
                                    print(f"\n📍 Objeto {obj_counter} → Punto malla M{closest_mesh_idx}:")
                                    print(f"   X = {X_m:.3f} m, Y = {Y_m:.3f} m, Z = {Z_m:.3f} m")

                        obj_counter += 1

                    cv2.imshow("Profundidad en zonas con diferencia", heatmap)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Profundidad en zonas con diferencia")

                captured_i = False
                img_i = None
                img_f = None
            else:
                print("\nError al capturar imagen con 'f'")
    elif key == ord('v'):
        show_depth_heatmap()

capL.release()
capR.release()
cv2.destroyAllWindows()
print("\nPrograma terminado.")