import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

"""Software dessarollado por CHATGPT a solicitud de mis instrucciones.
Script para analizar y mostrar histogramas de color (BGR y HSV)"""

def apply_red_mask(frame):
    """Aplica el mismo filtro rojo usado en tu código para la punta del robot."""
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
    return mask, hsv

def show_histogram_matplotlib(frame, mask, hsv):
    """Muestra histogramas con matplotlib."""
    red_pixels_bgr = cv2.bitwise_and(frame, frame, mask=mask)
    red_pixels_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

    non_zero_bgr = red_pixels_bgr[mask > 0]
    non_zero_hsv = red_pixels_hsv[mask > 0]

    if non_zero_bgr.size == 0:
        print("⚠️ No hay píxeles rojos detectados.")
        return

    plt.figure(figsize=(15, 4))

    # BGR histogram
    plt.subplot(1, 3, 1)
    plt.hist(non_zero_bgr[:, 0], bins=50, color='b', alpha=0.6, label='B')
    plt.hist(non_zero_bgr[:, 1], bins=50, color='g', alpha=0.6, label='G')
    plt.hist(non_zero_bgr[:, 2], bins=50, color='r', alpha=0.6, label='R')
    plt.title("Histograma BGR (rojos detectados)")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()

    # Hue (H)
    plt.subplot(1, 3, 2)
    plt.hist(non_zero_hsv[:, 0], bins=180, range=(0, 180), color='purple', alpha=0.7)
    plt.title("Matiz (H) - Rangos rojos: 0–6 y 174–180")
    plt.xlabel("H (0–180)")
    plt.ylabel("Frecuencia")

    # S y V
    plt.subplot(1, 3, 3)
    plt.hist(non_zero_hsv[:, 1], bins=256, range=(0, 255), color='gray', alpha=0.6, label='S')
    plt.hist(non_zero_hsv[:, 2], bins=256, range=(0, 255), color='orange', alpha=0.6, label='V')
    plt.title("Saturación (S) y Brillo (V)")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()

    plt.tight_layout()
    plt.show()

def draw_histogram_opencv(hist, color=(255, 255, 255), hist_height=300, hist_width=256):
    """Dibuja un histograma en una imagen de OpenCV."""
    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
    bin_w = int(round(hist_width / hist.shape[0]))
    for i in range(hist.shape[0]):
        cv2.line(hist_img,
                 (bin_w * i, hist_height),
                 (bin_w * i, hist_height - int(hist[i])),
                 color, thickness=1)
    return hist_img

def show_histogram_opencv(frame, mask, hsv):
    """Muestra histogramas usando solo OpenCV."""
    red_bgr = cv2.bitwise_and(frame, frame, mask=mask)
    red_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

    if cv2.countNonZero(mask) == 0:
        print("⚠️ No hay píxeles rojos detectados.")
        return

    # Histogramas de BGR (canales separados)
    hist_b = cv2.calcHist([red_bgr], [0], mask, [256], [0, 256])
    hist_g = cv2.calcHist([red_bgr], [1], mask, [256], [0, 256])
    hist_r = cv2.calcHist([red_bgr], [2], mask, [256], [0, 256])

    # Histograma de H (0–180)
    hist_h = cv2.calcHist([red_hsv], [0], mask, [180], [0, 180])
    # S y V (0–256)
    hist_s = cv2.calcHist([red_hsv], [1], mask, [256], [0, 256])
    hist_v = cv2.calcHist([red_hsv], [2], mask, [256], [0, 256])

    hist_b_img = draw_histogram_opencv(hist_b, color=(255, 0, 0))
    hist_g_img = draw_histogram_opencv(hist_g, color=(0, 255, 0))
    hist_r_img = draw_histogram_opencv(hist_r, color=(0, 0, 255))
    hist_h_img = draw_histogram_opencv(hist_h, color=(200, 50, 200), hist_width=180)
    hist_s_img = draw_histogram_opencv(hist_s, color=(150, 150, 150))
    hist_v_img = draw_histogram_opencv(hist_v, color=(0, 165, 255))

    top_row = np.hstack((hist_b_img, hist_g_img, hist_r_img))
    mid_row = np.hstack((hist_h_img, np.zeros_like(hist_h_img), np.zeros_like(hist_h_img)))
    mid_row = cv2.resize(mid_row, (top_row.shape[1], top_row.shape[0]))
    bot_row = np.hstack((hist_s_img, hist_v_img, np.zeros_like(hist_s_img)[:, :256]))

    h = top_row.shape[0]
    bot_row = cv2.resize(bot_row, (top_row.shape[1], h))

    full_hist = np.vstack((top_row, mid_row, bot_row))

    cv2.imshow("Histograma OpenCV (BGR + HSV)", full_hist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = "C:\\ARIS\\universidad\\TFG\\Programa\\experimentacionCNN\\pruebas\\capturas\\base.jpg"

    if not os.path.isfile(image_path):
        print(f"❌ Error: No se encontró '{image_path}'.")
        print("Coloca una imagen llamada 'imagen.jpg' en la misma carpeta o cambia 'image_path'.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error al cargar la imagen.")
        return

    print("🔍 Aplicando filtro rojo...")
    mask, hsv = apply_red_mask(frame)

    # Mostrar máscara y original con overlay (opcional, útil para depurar)
    red_only = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Original", frame)
    cv2.imshow("Máscara Roja", mask)
    cv2.imshow("Píxeles Rojos Detectados", red_only)
    cv2.waitKey(1000)  # Mostrar 1 segundo antes del histograma

    # Opción 1: Matplotlib
    print("📊 Generando histograma con Matplotlib...")
    show_histogram_matplotlib(frame, mask, hsv)

    # Opción 2: OpenCV
    print("📊 Generando histograma con OpenCV...")
    show_histogram_opencv(frame, mask, hsv)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()