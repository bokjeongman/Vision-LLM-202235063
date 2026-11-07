# 카메라 프레임의 YOLO 탐지 결과에서 화면 중앙 물건 이벤트를 감지하여 JSON 파일로 저장
#
# 중앙 물건 이벤트
# - 학습자가 배우고 싶은 물건을 카메라 화면 중앙에 보여주면 학습 시작 신호로 판단
# - 화면 중앙 영역(CENTER_RATIO) 안의 물건만 대상 (주변 배경 물건 제외, 사람 제외)
# - 같은 물건이 HOLD_FRAMES 동안 연속으로 유지되면 이벤트 발생 (스쳐 지나가는 물건 제외)
# - 이벤트 발생 시 JSON의 "event"에 물건 이름 저장 → llm.py가 읽어서 학습 진행

import cv2
import json
import os

from ultralytics import YOLO


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
JSON_PATH = "src/json/vision_data.json"
TEMP_JSON_PATH = "src/json/vision_data_tmp.json"

CENTER_RATIO = 0.5          # 화면 중앙 영역 비율
HOLD_FRAMES = 15            # 이벤트 판단 유지 프레임 수


# 화면 중앙에 있는 물건의 클래스 이름 반환
def check_event(objects, width, height):
    # 화면 중앙 영역 좌표
    center_x1 = width * (1 - CENTER_RATIO) / 2
    center_y1 = height * (1 - CENTER_RATIO) / 2
    center_x2 = width - center_x1
    center_y2 = height - center_y1

    candidates = []

    for obj in objects:
        # 사람은 제외
        if obj["class"] == "person":
            continue

        # BBox 중심점 계산
        bbox = obj["bbox"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2

        # 중심점이 중앙 영역 안에 있으면 후보로 추가
        if center_x1 <= cx <= center_x2 and center_y1 <= cy <= center_y2:
            candidates.append(obj)

    if len(candidates) == 0:
        return None

    # 후보 중 BBox 면적이 가장 큰 물건 선택
    target = None
    max_area = 0

    for obj in candidates:
        bbox = obj["bbox"]
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])

        if area > max_area:
            max_area = area
            target = obj

    return target["class"]


yolo = YOLO(YOLO_MODEL_PATH)

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
    )

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 화면 중앙 물건으로 강제 이벤트")

# 중앙 물건과 유지 프레임 수 저장
hold_class = None
hold_count = 0

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break

    height, width = frame.shape[:2]


    # 1. 현재 프레임 YOLO 탐지
    results = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        verbose=False,
    )

    result = results[0]


    # 2. YOLO 결과 화면 출력
    output_frame = result.plot()
    cv2.imshow("YOLO + Gemma", output_frame)


    # 3. YOLO 탐지 결과를 Dictionary 형태로 변환
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        objects.append(
            {
                "class": result.names[class_id],
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            }
        )

    # 4. 화면 중앙 물건이 HOLD_FRAMES 동안 유지되는지 확인
    center_class = check_event(objects, width, height)

    # 같은 물건이면 프레임 수 증가, 바뀌면 다시 측정
    if center_class == hold_class:
        hold_count = hold_count + 1
    else:
        hold_class = center_class
        hold_count = 1

    # HOLD_FRAMES 도달 또는 l 키 입력 시 이벤트 발생 (중앙 물건이 없으면 제외)
    event = False

    if hold_count == HOLD_FRAMES:
        event = True
    if key == ord("l"):
        event = True
    if center_class is None:
        event = False

    vision_dict = {
        "image_width": width,
        "image_height": height,
        "objects": objects,
        "event": center_class,
    }


    # 5. 이벤트 발생 시 Dictionary → JSON 파일 저장
    if event:
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(vision_dict, file, ensure_ascii=False, indent=4)

        # 6. 임시 파일 작성 완료 후 실제 파일로 교체
        os.replace(TEMP_JSON_PATH, JSON_PATH)

        print(f"\n[이벤트 감지] {center_class}")


cap.release()
cv2.destroyAllWindows()