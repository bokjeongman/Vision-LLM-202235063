# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
import cv2
import numpy as np

# ============================================================
# 1. 카메라 설정 (NVIDIA Jetson Nano / GStreamer)
# ============================================================
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
    raise RuntimeError(
        "카메라를 열 수 없습니다. GStreamer 설정과 /dev/video0을 확인하세요."
    )

# ============================================================
# 2. 색상 분할 설정 (LAB 색 공간)
# ============================================================
# 기존 코드의 임계값 유지
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# 최소 면적: 작은 잡음 Contour 제거
MIN_AREA = 500

# Morphological Opening용 타원형 커널
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

# ============================================================
# 3. Kalman Filter 설정
# 상태 변수: [x, y, vx, vy]
# 측정 변수: [x, y]
# ============================================================
kalman = cv2.KalmanFilter(4, 2)

# 상태 전이 행렬: 이전 위치와 속도로 다음 위치 예측
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# 측정 행렬: 위치 x, y만 관측
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

# 프로세스 잡음 공분산
kalman.processNoiseCov = np.diag([
    1e-2, 1e-2, 5e-2, 5e-2
]).astype(np.float32)

# 측정 잡음 공분산
kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 1.0
)

# 초기 오차 공분산
kalman.errorCovPost = np.eye(4, dtype=np.float32)

tracking = False
lost_frames = 0
MAX_LOST_FRAMES = 15
last_radius = 0

# ============================================================
# 4. 실시간 객체 검출 및 추적
# ============================================================
try:
    while True:

        # 프레임은 루프마다 한 번만 읽기
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # ----------------------------------------------------
        # (1) 이미지 전처리
        # OpenCV 영상은 BGR 형식
        # ----------------------------------------------------
        blurred = cv2.GaussianBlur(
            frame, (7, 7), 0
        )

        # BGR -> LAB
        lab = cv2.cvtColor(
            blurred, cv2.COLOR_BGR2LAB
        )

        # ----------------------------------------------------
        # (2) Color Segmentation: Binary Mask 생성
        # ----------------------------------------------------
        mask = cv2.inRange(
            lab, green_lower, green_upper
        )

        # ----------------------------------------------------
        # (3) Morphological Operation
        # Opening: 작은 잡음 제거
        # ----------------------------------------------------
        final_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=3
        )

        # ----------------------------------------------------
        # (4) Contour 검출 및 가장 큰 객체 선택
        # ----------------------------------------------------
        contours, _ = cv2.findContours(
            final_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected = False
        centroid = None
        largest_contour = None

        if len(contours) > 0:

            largest_contour = max(
                contours, key=cv2.contourArea
            )

            area = cv2.contourArea(largest_contour)

            if area >= MIN_AREA:

                detected = True

                # 최소 외접원: 중심과 반지름
                (x, y), radius = cv2.minEnclosingCircle(
                    largest_contour
                )

                center_circle = (int(x), int(y))
                radius = int(radius)

                # Moments를 이용한 무게중심 계산
                M = cv2.moments(largest_contour)

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = center_circle

                centroid = (cX, cY)

                # 검출된 객체의 중심과 외접원 (빨간색)
                cv2.circle(
                    frame, centroid, max(radius, 1),
                    (0, 0, 255), 2
                )
                cv2.circle(
                    frame, centroid, 5,
                    (0, 0, 255), -1
                )

                # 가장 큰 Contour 내부를 초록색으로 표시
                object_mask = np.zeros_like(final_mask)
                cv2.drawContours(
                    object_mask,
                    [largest_contour],
                    -1, 255, thickness=cv2.FILLED
                )

                green_overlay = np.zeros_like(frame)
                green_overlay[:, :, 1] = object_mask

                frame = cv2.addWeighted(
                    frame, 1.0,
                    green_overlay, 0.25, 0
                )

                cv2.drawContours(
                    frame, [largest_contour],
                    -1, (0, 255, 0), 2
                )

        # ----------------------------------------------------
        # (5) Kalman Filter: Prediction & Correction
        # ----------------------------------------------------
        if detected:

            measurement = np.array([
                [centroid[0]],
                [centroid[1]]
            ], dtype=np.float32)

            if not tracking:
                # 최초 검출 시 위치와 속도 초기화
                kalman.statePost = np.array([
                    [centroid[0]],
                    [centroid[1]],
                    [0.0],
                    [0.0]
                ], dtype=np.float32)

                kalman.statePre = kalman.statePost.copy()
                tracking = True

                state = kalman.statePost.copy()

            else:
                # 예측 후 실제 측정값으로 보정
                kalman.predict()
                state = kalman.correct(measurement)

            lost_frames = 0
            last_radius = radius

        elif tracking:

            # 객체가 잠시 가려졌다면 예측만 수행
            state = kalman.predict()
            lost_frames += 1

            # 너무 오래 검출되지 않으면 추적 초기화
            if lost_frames > MAX_LOST_FRAMES:
                tracking = False
                lost_frames = 0

        # ----------------------------------------------------
        # (6) 예측 위치 Overlay
        # ----------------------------------------------------
        if tracking:
            x_bel = int(state[0, 0])
            y_bel = int(state[1, 0])

            # 화면 밖 좌표를 방지
            h, w = frame.shape[:2]
            x_bel = np.clip(x_bel, 0, w - 1)
            y_bel = np.clip(y_bel, 0, h - 1)

            # Kalman Filter 추정 위치 (노란색)
            cv2.circle(
                frame,
                (int(x_bel), int(y_bel)),
                max(last_radius, 1),
                (0, 255, 255), 2
            )
            cv2.circle(
                frame,
                (int(x_bel), int(y_bel)),
                5,
                (0, 255, 255), -1
            )

        # ----------------------------------------------------
        # (7) 결과 표시
        # ----------------------------------------------------
        status = (
            "Detected" if detected
            else "Predicting" if tracking
            else "Not detected"
        )

        cv2.putText(
            frame, f"Status: {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        cv2.imshow("Basketball Detection", frame)
        cv2.imshow("Binary Mask", final_mask)

        # q 키로 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
