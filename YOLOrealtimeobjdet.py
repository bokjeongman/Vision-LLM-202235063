# from ultralytics import YOLO
# import cv2


# model = YOLO("src/models/YOLO/yolo11n.pt")
# model.to("cuda")

# pipeline = (
#     "v4l2src device=/dev/video0 ! "
#     "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
#     "videoconvert ! video/x-raw,format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 "
#     "max-size-bytes=0 max-size-time=0 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )

# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     results = model.predict(
#         source=frame,   # source image
#         conf=0.25,      # Confidence Threshold
#         iou=0.5,        # IoU Threshold
#         verbose=False,  # no output prints
#         classes=None,   # selected class
#     )

#     output_frame = results[0].plot()

#     cv2.imshow("YOLO Object Detection", output_frame)

# cap.release()
# cv2.destroyAllWindows()

from pathlib import Path
import yaml

import cv2
from ultralytics import YOLO


CALIBRATION_DIR = Path("src/datasets/calibration")
IMAGE_DIR = CALIBRATION_DIR / "images"
LABEL_DIR = CALIBRATION_DIR / "labels"
YAML_PATH = CALIBRATION_DIR / "calibration.yaml"

NUM_IMAGES = 500
SAVE_EVERY_N_FRAMES = 5

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


frame_count = 0
save_count = 0

while save_count < NUM_IMAGES:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Calibration: {save_count}/{NUM_IMAGES}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Calibration Image Collection", display_frame)

    if frame_count % SAVE_EVERY_N_FRAMES == 0:
        # Image (jpg) 파일 저장
        image_path = IMAGE_DIR / f"{save_count:04d}.jpg"
        cv2.imwrite(str(image_path), frame)

        # Label (txt) 파일 저장 (빈 파일)
        label_path = LABEL_DIR / f"{save_count:04d}.txt"
        label_path.touch()

        save_count += 1

        print(f"Saved: {save_count}/{NUM_IMAGES}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()


if save_count < NUM_IMAGES:
    raise RuntimeError(f"이미지가 {save_count}장만 저장되었습니다.")


model = YOLO("src/models/YOLO/yolo11n.pt")

# YAML 파일에 저장할 Dictionary
calibration_yaml = {
    "path": str(CALIBRATION_DIR.resolve()),
    "train": "images",
    "val": "images",
    "names": model.names,
}

# YAML 파일 저장
with open(YAML_PATH, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        calibration_yaml,
        file,
        sort_keys=False,
        allow_unicode=True,
    )

print()
print("Calibration Dataset 생성 완료")
print(f"Images: {IMAGE_DIR}")
print(f"YAML:   {YAML_PATH}")