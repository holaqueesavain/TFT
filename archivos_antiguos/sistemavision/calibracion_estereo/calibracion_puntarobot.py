import cv2
import numpy as np

# ----------------------------------------------------
# ---- Calibrador del color de la punta del robot ----
# ----------------------------------------------------
"""Software para calibrar la detección de la punta del robot mediante ajuste 
interactivo de parámetros HSV y morfológicos."""

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No se pudo abrir la cámara (índice 1).")
    exit()

cv2.namedWindow("Calibrador de Punta Robot")

cv2.createTrackbar("Modo Rojo Completo", "Calibrador de Punta Robot", 1, 1, lambda x: None)  # 0=Simple, 1=Doble

# HSV bajo (en caso de usar solo un umbral)
cv2.createTrackbar("H Min", "Calibrador de Punta Robot", 2, 179, lambda x: None)
cv2.createTrackbar("S Min", "Calibrador de Punta Robot", 160, 255, lambda x: None)
cv2.createTrackbar("V Min", "Calibrador de Punta Robot", 170, 255, lambda x: None)

# HSV alto (en caso de usar dos umbrales)
cv2.createTrackbar("H Max", "Calibrador de Punta Robot", 9, 179, lambda x: None)
cv2.createTrackbar("S Max", "Calibrador de Punta Robot", 220, 255, lambda x: None)
cv2.createTrackbar("V Max", "Calibrador de Punta Robot", 240, 255, lambda x: None)

cv2.createTrackbar("Min Area", "Calibrador de Punta Robot", 50, 2000, lambda x: None)

cv2.createTrackbar("AR Min (x100)", "Calibrador de Punta Robot", 30, 300, lambda x: None)
cv2.createTrackbar("AR Max (x100)", "Calibrador de Punta Robot", 200, 500, lambda x: None)

print("Ajusta los parámetros hasta que solo la punta del robot esté detectada.")
print("'Modo Rojo Completo = 1' activa detección en ambos extremos del rojo (0° y 180°).")
print("Presiona 'q' para salir. Los parámetros finales se imprimirán al salir.")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    modo_rojo_completo = cv2.getTrackbarPos("Modo Rojo Completo", "Calibrador de Punta Robot")
    h_min = cv2.getTrackbarPos("H Min", "Calibrador de Punta Robot")
    s_min = cv2.getTrackbarPos("S Min", "Calibrador de Punta Robot")
    v_min = cv2.getTrackbarPos("V Min", "Calibrador de Punta Robot")
    h_max = cv2.getTrackbarPos("H Max", "Calibrador de Punta Robot")
    s_max = cv2.getTrackbarPos("S Max", "Calibrador de Punta Robot")
    v_max = cv2.getTrackbarPos("V Max", "Calibrador de Punta Robot")

    min_area = cv2.getTrackbarPos("Min Area", "Calibrador de Punta Robot")
    ar_min_x100 = cv2.getTrackbarPos("AR Min (x100)", "Calibrador de Punta Robot")
    ar_max_x100 = cv2.getTrackbarPos("AR Max (x100)", "Calibrador de Punta Robot")
    ar_min = ar_min_x100 / 100.0
    ar_max = ar_max_x100 / 100.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    if modo_rojo_completo == 1:
        # valores cercanos a 0
        lower1 = np.array([0, s_min, v_min])
        upper1 = np.array([h_max, s_max, v_max])  # h_max suele ser pequeño

        # valores cercanos a 180
        lower2 = np.array([180 - h_max, s_min, v_min])  # ej 170
        upper2 = np.array([180, s_max, v_max])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)   
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close) 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        if not (ar_min <= aspect_ratio <= ar_max):
            continue

        cv2.drawContours(output, [cnt], -1, (0, 255, 255), 2)
        center = (x + w // 2, y + h // 2)
        cv2.circle(output, center, 5, (0, 0, 255), -1)
        cv2.putText(output, "Punta robot", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        detected = True

    cv2.imshow("Original", output)
    cv2.imshow("Mascara", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n" + "="*60)
        if modo_rojo_completo == 1:
            print("# Modo Rojo Completo ACTIVADO: usa DOS rangos HSV")
            print(f"# Rango 1 (bajo):  np.array([0, {s_min}, {v_min}]) → np.array([{h_max}, {s_max}, {v_max}])")
            print(f"# Rango 2 (alto): np.array([{180 - h_max}, {s_min}, {v_min}]) → np.array([180, {s_max}, {v_max}])")
        else:
            print(f"lower_color = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_color = np.array([{h_max}, {s_max}, {v_max}])")
        print(f"min_area = {min_area}")
        print(f"aspect_ratio_range = ({ar_min:.2f}, {ar_max:.2f})")
        print("="*60)
        break

cap.release()
cv2.destroyAllWindows()