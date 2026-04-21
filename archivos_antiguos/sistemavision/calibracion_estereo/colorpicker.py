import cv2
import numpy as np

def click_event(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        b, g, r = current_frame[y, x]
        
        bgr_pixel = np.uint8([[[b, g, r]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0, 0]
        
        print(f"Clic en ({x}, {y}) → HSV (OpenCV) = (H={h}, S={s}, V={v})")
        
        color_patch = np.full((100, 100, 3), (b, g, r), dtype=np.uint8)
        color_patch_rgb = cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB)
        cv2.imshow("Color clickeado", color_patch_rgb)

cap = cv2.VideoCapture(1)
current_frame = None

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

cv2.namedWindow("RGB/HSV Inspector")
cv2.setMouseCallback("RGB/HSV Inspector", click_event)

print("Haz clic en cualquier punto para ver su color RGB y HSV.")
print("Presiona 'q' para salir.")

while True:
    ret, current_frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    cv2.imshow("RGB/HSV Inspector", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()