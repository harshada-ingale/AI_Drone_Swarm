import cv2, time

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(i, "cannot open")
        continue
    time.sleep(1.0)
    for _ in range(30):                  # let auto-exposure settle
        ok, frame = cap.read()
    if ok:
        print(i, "brightness:", round(float(frame.mean()), 1), "size:", frame.shape[:2])
        cv2.imwrite(f"cam_{i}.jpg", frame)
    else:
        print(i, "opened but no frame")
    cap.release()