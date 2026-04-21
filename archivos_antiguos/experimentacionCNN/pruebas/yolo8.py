import cv2
import numpy as np
import os
from ultralytics import YOLO
import time

inicio_ejecucion = time.time()

print("Cargando YOLOv8n...")
model = YOLO('C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\modelos\\yolov8n.pt')

FRUIT_CLASSES = {
    'apple', 'banana', 'orange'
}

def get_diff_mask(bg_img, curr_img):
    gray_bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_bg, gray_curr)
    umbral_dif = 25
    mask = np.uint8(diff > umbral_dif) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def process_image(bg_img, curr_img):
    mask_diff = get_diff_mask(bg_img, curr_img)
    
    results = model(curr_img, conf=0.25, iou=0.45, imgsz=640, verbose=False)
    
    result_img = curr_img.copy()
    valid_detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id].lower()
        conf = float(box.conf[0].cpu().numpy())
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if cls_name not in FRUIT_CLASSES:
            continue

        roi_mask = mask_diff[y1:y2, x1:x2]
        if cv2.countNonZero(roi_mask) < 50:  # mínimo de píxeles con diferencia
            continue

        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(result_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        valid_detections.append(label)

    label_text = ", ".join(valid_detections) if valid_detections else "No frutas válidas"
    return result_img, label_text

def build_mosaic(images, labels):
    H, W = images[0].shape[:2]
    cell_h = H + 40
    cell_w = W
    mosaic = np.ones((cell_h * 3, cell_w * 3, 3), dtype=np.uint8) * 255
    k = 0
    for r in range(3):
        for c in range(3):
            img = images[k]
            txt = labels[k]
            mosaic[r*cell_h:r*cell_h+H, c*cell_w:c*cell_w+W] = img
            cv2.putText(mosaic, txt[:60], (c*cell_w + 10, r*cell_h + H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            k += 1
    return mosaic

bg_img = cv2.imread("C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\pruebas\\capturas\\base.jpg")
if bg_img is None:
    raise FileNotFoundError("'base.jpg' no encontrado o inválido")

image_paths = [f"C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\pruebas\\capturas\\Img{i}.jpg" for i in range(2, 11)]
processed_imgs = []
labels_list = []

for path in image_paths:
    print(f"Procesando {path}...")
    curr_img = cv2.imread(path)
    if curr_img is None:
        curr_img = np.zeros_like(bg_img) # si no hay img entonces se usa negro
        result_img = curr_img.copy()
        label_txt = "Imagen no encontrada"
    else:
        if curr_img.shape != bg_img.shape:
            curr_img = cv2.resize(curr_img, (bg_img.shape[1], bg_img.shape[0]))
        result_img, label_txt = process_image(bg_img, curr_img)

    processed_imgs.append(result_img)
    labels_list.append(label_txt)

mosaic = build_mosaic(processed_imgs, labels_list)
cv2.imwrite("output\\mosaico_yolo_8n.jpg", mosaic)
print("Guardado: output\\mosaico_yolo_8n.jpg")

# printear el tiempo
duracion_total = time.time() - inicio_ejecucion
horas = int(duracion_total // 3600)
minutos = int((duracion_total % 3600) // 60)
segundos = int(duracion_total % 60)
print(f"\nDuración: {horas:02d}:{minutos:02d}:{segundos:02d}")