from ultralytics import YOLO
import cv2
import time


model = YOLO("src/models/YOLO/yolo11n.pt")
model.to("cuda")

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

displayed_fps = 0.0

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    results = model.predict(
        source=frame,   # source image
        conf=0.25,      # Confidence Threshold
        iou=0.5,        # IoU Threshold
        verbose=False,  # no output prints
        classes=None,   # selected class
    )

    output_frame = results[0].plot()

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    cv2.putText(output_frame, f"FPS: {displayed_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection with FPS", output_frame)

cap.release()
cv2.destroyAllWindows()
