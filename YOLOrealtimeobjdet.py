from ultralytics import YOLO
import cv2


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

while True:
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

    cv2.imshow("YOLO Object Detection", output_frame)

cap.release()
cv2.destroyAllWindows()
