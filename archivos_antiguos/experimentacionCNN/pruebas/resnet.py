import cv2
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time

inicio_ejecucion = time.time()

print("Cargando ResNet50...")
modelo = ResNet50(weights='imagenet', include_top=True)

fruit_classes = {
    'banana', 'orange', 'lemon', 'pineapple', 'watermelon', 'mango',
    'apple', 'pear', 'grape', 'strawberry', 'pomegranate', 'coconut'
}

def classify_fruit_from_roi(roi):
    if roi.size == 0 or min(roi.shape[:2]) < 20:
        return None, 0.0
    roi_resized = cv2.resize(roi, (224, 224))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_input = preprocess_input(np.expand_dims(roi_rgb, axis=0))
    preds = modelo.predict(roi_input, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    for _, class_name, prob in decoded:
        if class_name in fruit_classes:
            return class_name, prob
    return None, 0.0

def process_image(bg_img, curr_img):
    gray_bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_bg, gray_curr)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = curr_img.copy()
    labels = []

    if not contours:
        return result, "No hay objetos"

    detection_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 600:
            continue

        detection_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(curr_img.shape[1] - x, w + 2 * margin)
        h = min(curr_img.shape[0] - y, h + 2 * margin)

        roi = curr_img[y:y+h, x:x+w]
        fruit, prob = classify_fruit_from_roi(roi)

        color = (0, 255, 0) if fruit else (0, 0, 255)
        label = f"{fruit} ({prob:.2f})" if fruit else "No fruta"
        labels.append(label)

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result, ", ".join(labels) if labels else "No frutas detectadas"

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
    raise FileNotFoundError("'base.jpg' no encontrado")

image_paths = [f"C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\pruebas\\capturas\\Img{i}.jpg" for i in range(2, 11)]
processed_imgs = []
label_texts = []

for path in image_paths:
    print(f"Procesando {path}...")
    curr_img = cv2.imread(path)
    if curr_img is None:
        print(f"{path} no encontrado. Usando imagen negra.")
        curr_img = np.zeros_like(bg_img)
        result_img = curr_img.copy()
        label_txt = "Imagen no encontrada"
    else:
        if curr_img.shape != bg_img.shape:
            curr_img = cv2.resize(curr_img, (bg_img.shape[1], bg_img.shape[0]))
        result_img, label_txt = process_image(bg_img, curr_img)

    processed_imgs.append(result_img)
    label_texts.append(label_txt)

mosaic = build_mosaic(processed_imgs, label_texts)
cv2.imwrite("output\\mosaico_resnet50.jpg", mosaic)
print("Guardado: output\\mosaico_resnet50.jpg")

# printear el tiempo
duracion_total = time.time() - inicio_ejecucion
horas = int(duracion_total // 3600)
minutos = int((duracion_total % 3600) // 60)
segundos = int(duracion_total % 60)
print(f"\nDuración: {horas:02d}:{minutos:02d}:{segundos:02d}")