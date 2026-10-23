import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import platform # use for my notebook and NVIDIA Jetson Nano

if platform.system() == "Windows": # check OS for cv2.VideoCapture() function
    cap = cv2.VideoCapture(0) # use for my notebook
else:
    pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) # use again for NVIDIA Jetson Nano


# 중간 점 기준으로 각도 계산하는 함수
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보

finger_tips = (4, 8, 12, 16, 20)  # 손가락 끝 landmark의 index
angle_threshold = 160             # 손가락 펼쳐짐 여부 판단 각도 임계값

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3),      # 엄지: 2번 중심
    (5, 6, 7),      # 검지: 6번 중심
    (9, 10, 11),    # 중지: 10번 중심
    (13, 14, 15),   # 약지: 14번 중심
    (17, 18, 19),   # 소지: 18번 중심
)



while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 프레임 높이와 너비
    h, w = frame.shape[:2]

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 프레임 내 손 탐지
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_detector.detect(mp_image)

    # 좌우 반전된 화면을 기준으로 왼손과 오른손 정보 변경
    labels = ["Left" if handedness[0].category_name == "Right" else "Right" for handedness in result.handedness]

    # 감지된 모든 손에서 펼쳐진 손가락 개수 계산
    total_finger_count = 0
    for hand in result.hand_landmarks:
        for point1_idx, point2_idx, point3_idx in finger_angle_points:
            angle = calculate_angle(
                hand[point1_idx],
                hand[point2_idx],
                hand[point3_idx],
            )

            if angle >= angle_threshold:
                total_finger_count += 1

    # 화면 좌측 상단에 손 개수와 펼친 손가락 개수 표시
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Fingers: {total_finger_count}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # 화면 우측 상단에 왼손/오른손/양손 여부 표시
    handedness_text = " / ".join(labels)
    text_size = cv2.getTextSize(handedness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = w - text_size[0] - 20
    cv2.putText(frame, handedness_text, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # 탐지 결과의 각 손마다 선과 점 그리기
    for hand in result.hand_landmarks:
        h, w = frame.shape[:2]  # 프레임 높이와 너비
        points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

        # landmark를 연결하는 선 (skeleton) 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)

        # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
        for i, point in enumerate(points):
            color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

    cv2.imshow("MediaPipe Hand Detection", frame)

hand_detector.close()
cap.release()
cv2.destroyAllWindows()