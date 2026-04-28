"""
Ajuste de los parámetros SGBM y la matriz Q para la profundidad. 
Sirve para hallar el valor 'f' (focal) y el 'B' (baseline) en metros 
para que la distancia Z sea lo más real posible.
"""

import cv2
import numpy as np

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)

if not capL.isOpened() or not capR.isOpened():
    print("Cámaras no disponibles")
    exit()

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

focal_px = 980        
baseline_mm = 93    
Q = None
disp_map = None

def update_Q(f, B):
    cx, cy = 320, 240 
    return np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0,  f],
        [0, 0, -1.0 / B, 0]
    ])

def point_to_3d(x, y, disp_val, Q):
    if disp_val <= 0: return None
    point = np.dot(Q, np.array([x, y, disp_val, 1.0], dtype=np.float32))
    return point[:3] / point[3]

cv2.namedWindow("Ajuste Q en vivo")
cv2.createTrackbar("Focal (px)", "Ajuste Q en vivo", focal_px, 10000, lambda x: None)
cv2.createTrackbar("Baseline (mm)", "Ajuste Q en vivo", baseline_mm, 500, lambda x: None) 

print("Ajusta focal y baseline en tiempo real. Presiona 'q' para salir.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR: break

    focal_px = cv2.getTrackbarPos("Focal (px)", "Ajuste Q en vivo")
    baseline_mm = cv2.getTrackbarPos("Baseline (mm)", "Ajuste Q en vivo")

    if focal_px < 1: focal_px = 1
    if baseline_mm < 1: baseline_mm = 1
    baseline_m = baseline_mm / 1000.0

    Q = update_Q(focal_px, baseline_m)

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    disp_raw = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_raw[disp_raw <= 0] = 0.1

    disp_map = cv2.medianBlur(disp_raw, 5)
    disp_map = cv2.GaussianBlur(disp_map, (5, 5), 0)

    d_center = disp_map[240, 320]
    xyz = point_to_3d(320, 240, d_center, Q)

    vis = frameL.copy()
    cv2.circle(vis, (320, 240), 8, (0, 0, 255), -1)
    if xyz is not None:
        text = f"X:{xyz[0]:.3f} Y:{xyz[1]:.3f} Z:{xyz[2]:.3f} m"
        cv2.putText(vis, text, (10, 30), 0, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, f"f={focal_px}px, B={baseline_m:.3f}m", (10, 60), 0, 0.6, (255, 255, 255), 2)

    cv2.imshow("Ajuste Q en vivo", vis)

    with np.errstate(divide='ignore'):
        depth_map = (focal_px * baseline_m) / (disp_map + 1e-6)
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow("Depth Heatmap", depth_colored)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

capL.release()
capR.release()
cv2.destroyAllWindows()