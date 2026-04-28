"""
Herramienta para alinear físicamente las cámaras. Superpone la imagen 
izquierda (verde) y derecha (roja) para ajustar el ángulo y el foco.
"""

import cv2
import numpy as np

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOCUS, 0)

if not capL.isOpened() or not capR.isOpened():
    print("Cámaras no disponibles")
    exit()

cv2.namedWindow("1. Alineación (G=Izq, R=Der)")
cv2.namedWindow("2. Disparidad (Mapa de calor)")
cv2.namedWindow("3. Controles")

control_img = np.zeros((100, 400, 3), dtype=np.uint8)
cv2.imshow("3. Controles", control_img)
cv2.createTrackbar("minDisp", "3. Controles", 0, 50, lambda x: None)
cv2.createTrackbar("numDisp", "3. Controles", 6, 20, lambda x: None)  # x16
cv2.createTrackbar("blockSize", "3. Controles", 11, 21, lambda x: None)
cv2.createTrackbar("uniqueness", "3. Controles", 10, 20, lambda x: None)
cv2.createTrackbar("speckleWin", "3. Controles", 100, 200, lambda x: None)
cv2.createTrackbar("speckleRange", "3. Controles", 32, 50, lambda x: None)
cv2.createTrackbar("P1", "3. Controles", 200, 1000, lambda x: None)
cv2.createTrackbar("P2", "3. Controles", 800, 4000, lambda x: None)

print("Presiona 'q' para salir.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR: break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    overlay = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay[:, :, 1] = grayL 
    overlay[:, :, 2] = grayR 
    cv2.imshow("1. Alineación (G=Izq, R=Der)", overlay)

    min_disp = cv2.getTrackbarPos("minDisp", "3. Controles")
    num_disp = cv2.getTrackbarPos("numDisp", "3. Controles") * 16
    if num_disp < 16: num_disp = 16
    block_size = cv2.getTrackbarPos("blockSize", "3. Controles")
    if block_size % 2 == 0: block_size += 1
    uniqueness = cv2.getTrackbarPos("uniqueness", "3. Controles")
    speckle_win = cv2.getTrackbarPos("speckleWin", "3. Controles")
    speckle_range = cv2.getTrackbarPos("speckleRange", "3. Controles")
    P1 = cv2.getTrackbarPos("P1", "3. Controles")
    P2 = cv2.getTrackbarPos("P2", "3. Controles")

    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_win,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = 0.1
    disp = cv2.medianBlur(disp, 5)
    disp = cv2.GaussianBlur(disp, (5, 5), 0)

    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    h, w = disp.shape
    cx, cy = w // 2, h // 2
    d_center = disp[cy, cx]
    focal_px = 1000
    baseline_m = 0.093
    Z = (focal_px * baseline_m) / (d_center + 1e-6) if d_center > 0 else 0
    cv2.circle(heatmap, (cx, cy), 8, (255, 255, 255), 2)
    cv2.putText(heatmap, f"Z ≈ {Z:.2f} m", (10, 30), 0, 0.7, (255, 255, 255), 2)

    cv2.imshow("2. Disparidad (Mapa de calor)", heatmap)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

capL.release()
capR.release()
cv2.destroyAllWindows()