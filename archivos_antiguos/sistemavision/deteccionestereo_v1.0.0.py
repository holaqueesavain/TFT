import cv2
import numpy as np
import math
import time
import os
from ultralytics import YOLO

# ----------------------------------------------------
# --------------------- V.1.0.0 ----------------------
# ------------------ Vision estereo ------------------
# ------------------ + Red neuronal ------------------
# ----------------------------------------------------
"""Esta versión reemplaza la detección por diferencia de fondo 
con YOLOv11n. Solo se detectan frutas dentro del área de malla.
El punto más bajo del bounding box (centro X, base Y) se usa 
para estimar altura y vincular con el punto de malla más cercano."""

points = []          
mesh_points = []  
obj_center = None    
H = None             
H_inv = None         
rectified_width, rectified_height = 350, 200  

last_update_time = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_i = None      
img_f = None       
captured_i = False
capL = None
capR = None
stereo = None
Q = None  
disp_map = None  

manual_selection_mode = False

print("Cargando YOLOv11n...")
model = YOLO('C:\\ARIS\\universidad\\TFG\\Programa\\sistemavision\\yolo11n.pt')
FRUIT_CLASSES = {'apple', 'orange', 'banana'}

def load_calibration():
    global Q, stereo, capL, capR
    baseline = 0.093
    focal_length_px = 1100
    img_width, img_height = 640, 480
    cx = img_width / 2
    cy = img_height / 2

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

    if not capL.isOpened() or not capR.isOpened():
        print("Cámaras no disponibles")
        return False

    stereo = cv2.StereoSGBM_create(
        minDisparity=17,
        numDisparities=112,
        blockSize=7,
        P1=1000,
        P2=4000,
        disp12MaxDiff=1,
        uniquenessRatio=0,
        speckleWindowSize=60,
        speckleRange=12
    )
    return True

def get_disparity():
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

def setup_homography_and_mesh():
    global mesh_points, H, H_inv
    if len(points) != 4:
        return

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

    with open(os.path.join(OUTPUT_DIR, "puntos_mallado.txt"), "w") as f:
        for pt in mesh_points:
            f.write(f"{pt[0]} {pt[1]}\n")
    print("Archivo 'puntos_mallado.txt' actualizado.")

def is_point_in_mesh(x, y):
    """Verifica si (x,y) está dentro del polígono de los 4 puntos de malla."""
    if len(points) != 4:
        return False
    pts = np.array(points, dtype=np.float32)
    poly = np.array([pts[0], pts[1], pts[2], pts[3]], dtype=np.float32)
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

def get_bottom_center(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    cx = (x1 + x2) // 2
    cy = y2 
    return cx, cy

def select_points(event, x, y, flags, param):
    global points, manual_selection_mode
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Punto {len(points)}: ({x}, {y})")
        if len(points) == 4:
            setup_homography_and_mesh()
            manual_selection_mode = False

def initialize_with_default_points():
    global points, manual_selection_mode
    points = [
        [174, 291],
        [142, 419],
        [572, 440],
        [448, 306]
    ]
    manual_selection_mode = False
    setup_homography_and_mesh()
    print("Puntos por defecto cargados.")

def show_depth_heatmap():
    if disp_map is None:
        print("No hay disparidad calculada")
        return
    depth_normalized = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(depth_normalized, colormap=cv2.COLORMAP_JET)
    if len(mesh_points) > 0:
        for pt in mesh_points:
            cv2.circle(heatmap, pt, 3, (255, 255, 255), -1)
    cv2.imshow("Profundidad con Mesh", heatmap)
    cv2.waitKey(0)

if not load_calibration():
    exit()

cv2.namedWindow("Triangular Workspace")
initialize_with_default_points()

print("Iniciando bucle principal. Presiona 'q' para salir, 'r' para redefinir puntos.")

while True:
    ret, frame = capL.read()
    if not ret:
        break

    disp = get_disparity()

    if manual_selection_mode:
        for i, pt in enumerate(points):
            cv2.circle(frame, tuple(pt), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"P{i+1}", tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        for pt in mesh_points:
            cv2.circle(frame, pt, 3, (255, 0, 0), -1)

    # para una punta roja
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    s_min, v_min = 29, 137
    s_max, v_max = 255, 255
    h_max_low = 6
    h_min_high = 174

    lower1 = np.array([0, s_min, v_min])
    upper1 = np.array([h_max_low, s_max, v_max])
    lower2 = np.array([h_min_high, s_min, v_min])
    upper2 = np.array([180, s_max, v_max])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_center = None
    min_area = 27
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
    if current_time - last_update_time >= 0.2:
        with open(os.path.join(OUTPUT_DIR, "punta_robot.txt"), "w") as f:
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

    if manual_selection_mode:
        cv2.putText(frame, f"Selecciona punto {len(points)+1}/4 con clic izquierdo", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Área de trabajo definida. Presiona 'r' para redefinir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'v' para ver mapa disparidad", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status = "Listo para 'i'" if not captured_i else "Listo para 'f'"
    cv2.putText(frame, f"Estado: {status}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
        manual_selection_mode = True
        cv2.setMouseCallback("Triangular Workspace", select_points)
        with open(os.path.join(OUTPUT_DIR, "puntos_mallado.txt"), "w") as f:
            pass
        print("\nModo selección manual activado. Haz clic en 4 puntos.")

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
                print("\nImagen 'f' capturada. Detectando frutas con YOLO y calculando disparidad...")

                # calcular máscara de diferencia
                gray_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
                gray_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray_i, gray_f)
                umbral_dif = 25
                mask_diff_original = np.uint8(diff > umbral_dif) * 255
                kernel = np.ones((5, 5), np.uint8)
                mask_diff_processed = cv2.morphologyEx(mask_diff_original, cv2.MORPH_CLOSE, kernel, iterations=1)

                # calcular disparidad
                disp_map = get_disparity()
                if disp_map is None or disp_map.shape != mask_diff_original.shape:
                    print("Disparidad no disponible o tamaño incorrecto")
                    captured_i = False
                    continue

                # aplicar máscara a disparidad
                disp_masked = np.where(mask_diff_original > 0, disp_map, 0)
                disp_vis = cv2.normalize(disp_masked, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

                # dibujar mallado
                for pt in mesh_points:
                    cv2.circle(heatmap, pt, 2, (255, 255, 255), -1)

                # detectar todas las frutas con YOLO
                results = model(img_f, conf=0.25, iou=0.45, imgsz=640, verbose=False)
                fruit_boxes = []

                for box in results[0].boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if cls_name in FRUIT_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        fruit_boxes.append((x1, y1, x2, y2, cls_name))

                # filtrar solo frutas DENTRO del área de malla y con cambio detectado
                valid_detections = []
                for (x1, y1, x2, y2, cls_name) in fruit_boxes:
                    cx = (x1 + x2) // 2
                    cy = y2 

                    if not is_point_in_mesh(cx, cy):
                        continue

                    roi_mask = mask_diff_original[y1:y2, x1:x2]
                    if cv2.countNonZero(roi_mask) < 50:  # mínimo de píxeles con cambio
                        continue

                    valid_detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'cls_name': cls_name, 'cx': cx, 'cy': cy
                    })

                if len(valid_detections) == 0:
                    print("No se detectaron frutas válidas dentro del área de malla con cambio detectado")
                    with open(os.path.join(OUTPUT_DIR, "objetos.txt"), "w") as f_obj:
                        pass
                    captured_i = False
                    img_i = None
                    img_f = None
                    continue

                # mostrar resultados sobre el heatmap de disparidad
                obj_counter = 1
                detected_mesh_labels = []

                for det in valid_detections:
                    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                    cls_name = det['cls_name']
                    cx, cy = det['cx'], det['cy']

                    cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(heatmap, f"{cls_name} {obj_counter}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.circle(heatmap, (cx, cy), 5, (0, 0, 255), -1)

                    # estimacion de altura
                    xyz_bottom = point_to_3d(cx, cy)
                    altura_estimada = None

                    if xyz_bottom is not None:
                        X_obj, Y_obj, Z_obj = xyz_bottom
                        cv2.putText(heatmap, f"Z={Z_obj:.2f}m", (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        focal_length_px = 700
                        altura_pixeles = y2 - y1
                        altura_estimada = (altura_pixeles * Z_obj) / focal_length_px

                        if altura_estimada < 0.01 or altura_estimada > 0.20:
                            print(f"Altura no plausible ({altura_estimada:.3f} m). Objeto ignorado.")
                            altura_estimada = 0.20 #valor por defecto
                        else:
                            cv2.putText(heatmap, f"Altura={altura_estimada*100:.1f}cm", (x1, y2 + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if len(mesh_points) > 0:
                        min_dist = float('inf')
                        closest_mesh_idx = -1
                        for j, mesh_pt in enumerate(mesh_points):
                            dist = np.linalg.norm(np.array([cx, cy]) - np.array(mesh_pt))
                            if dist < min_dist:
                                min_dist = dist
                                closest_mesh_idx = j

                        if closest_mesh_idx != -1:
                            closest_pt = mesh_points[closest_mesh_idx]
                            cv2.circle(heatmap, closest_pt, 6, (255, 255, 0), -1)
                            cv2.putText(heatmap, f"P{closest_mesh_idx}", (closest_pt[0] + 5, closest_pt[1] + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                            if altura_estimada is not None:
                                label = f"p{closest_mesh_idx} {altura_estimada:.6f} {cls_name}"
                                detected_mesh_labels.append(label)
                        else:
                            print(f"No se encontró punto de malla cercano.")

                    obj_counter += 1

                with open(os.path.join(OUTPUT_DIR, "objetos.txt"), "w") as f_obj:
                    for label in detected_mesh_labels:
                        f_obj.write(label + "\n")
                print(f"Archivo 'objetos.txt' guardado con {len(detected_mesh_labels)} fruta(s).")

                cv2.imshow("Profundidad en zonas con diferencia (filtrada por YOLO)", heatmap)
                cv2.waitKey(0)
                cv2.destroyWindow("Profundidad en zonas con diferencia (filtrada por YOLO)")

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