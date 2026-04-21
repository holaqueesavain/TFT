import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import time

# para medir tiempo de ejecucion
inicio_ejecucion = time.time()

print("Cargando MobileNetV2...")
modelo = MobileNetV2(weights='imagenet', include_top=True)

fruit_classes = {
    'banana', 'orange',
    'apple'
}

def classify_fruit_from_roi(roi):
    if roi.size == 0:
        return None, 0.0
    roi_resized = cv2.resize(roi, (224, 224))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_array = np.expand_dims(roi_rgb, axis=0)
    roi_preprocessed = preprocess_input(roi_array)

    preds = modelo.predict(roi_preprocessed, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    for _, class_name, prob in decoded:
        if class_name == 'apple':
            return 'apple', prob
        if class_name in fruit_classes and prob > 0.10:
            return class_name, prob
    return None, 0.0

def detect_fruits_in_image(bg_frame, curr_frame):
    gray_bg = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_bg, gray_curr)

    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = curr_frame.copy()
    labels_list = []

    if not contours:
        return result, "No frutas detectadas"

    deteccion_id = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(curr_frame.shape[1] - x, w + 2 * margin)
        h = min(curr_frame.shape[0] - y, h + 2 * margin)

        roi = curr_frame[y:y+h, x:x+w].copy()
        fruit, prob = classify_fruit_from_roi(roi)

        color = (0, 255, 0) if fruit else (0, 0, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        label = f"{fruit} {prob:.2f}" if fruit else "No fruta"
        cv2.putText(result, f"{deteccion_id}: {label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        labels_list.append(label)
        deteccion_id += 1

    return result, ", ".join(labels_list) if labels_list else "No frutas detectadas"

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
            cv2.putText(mosaic, txt, (c*cell_w + 10, r*cell_h + H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            k += 1
    return mosaic

bg_frame = cv2.imread("C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\pruebas\\capturas\\base.jpg")
if bg_frame is None:
    print("Error: 'base.jpg' no encontrado o inválido.")
    exit()

img_paths = [f"C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\pruebas\\capturas\\Img{i}.jpg" for i in range(2, 11)]
final_images = []
final_labels = []

for path in img_paths:
    curr_frame = cv2.imread(path)
    if curr_frame is None:
        print(f"Advertencia: '{path}' no encontrado. Usando imagen negra.")
        h, w = bg_frame.shape[:2]
        curr_frame = np.zeros((h, w, 3), dtype=np.uint8)

    if curr_frame.shape != bg_frame.shape:
        curr_frame = cv2.resize(curr_frame, (bg_frame.shape[1], bg_frame.shape[0]))

    result_img, label_txt = detect_fruits_in_image(bg_frame, curr_frame)
    final_images.append(result_img)
    final_labels.append(label_txt)

mosaic = build_mosaic(final_images, final_labels)
cv2.imwrite("output\\mosaico_mobilenet.jpg", mosaic)
print("Guardado: output\\mosaico_mobilenet.jpg")

# printear el tiempo
duracion_total = time.time() - inicio_ejecucion
horas = int(duracion_total // 3600)
minutos = int((duracion_total % 3600) // 60)
segundos = int(duracion_total % 60)
print(f"\nDuración: {horas:02d}:{minutos:02d}:{segundos:02d}")