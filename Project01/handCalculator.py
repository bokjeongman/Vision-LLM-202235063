# This file is calculator ...

# import modules
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import platform # use for my notebook and NVIDIA Jetson Nano
import time # 제스처 유지 시간 측정용

# cv2.VideoCapture() function를 위한 OS 확인
if platform.system() == "Windows": # Windows 노트북 용
    cap = cv2.VideoCapture(0) 
else: # NVIDIA Jetson Nano에 쓰이는 pipeline
    pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) 


# 중간 점 기준으로 각도 계산하는 함수: 
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]) # p2p1 vector
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]) # p2p3 vector

    magnitude1 = np.linalg.norm(vector1) # 벡터 크기 구하기 (unit vector)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0: # 크기가 0일때 분모 0 되므로 예외 처리
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2) # 코사인 유사도 (cosine similarity) 계산 내적/(크기*크기)
    cosine = np.clip(cosine, -1.0, 1.0) # 코사인 유사도 값이 -1.0 ~ 1.0 범위를 벗어나지 않도록 제한 (clipping)

    return np.degrees(np.arccos(cosine)) # arccos로 각도 계산 후 radian -> degree 단위로 변환

# base_option부터 finger_tips까지가 baseline 코드
base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정: 손 갯수가 3개가 되지 않도록 함
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보

finger_tips = (4, 8, 12, 16, 20)  # 손가락 끝 landmark의 index (엄지, 검지, 중지, 약지, 소지)
angle_threshold = 160             # 손가락 펼쳐짐 여부 판단 각도 임계값

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = ( # 0번은 손목 landmark이므로 제외
    (1, 2, 3),      # 엄지: 2번 중심
    (5, 6, 7),      # 검지: 6번 중심
    (9, 10, 11),    # 중지: 10번 중심
    (13, 14, 15),   # 약지: 14번 중심
    (17, 18, 19),   # 소지: 18번 중심 
)

# ALU 딕셔너리 정의: 사칙연산을 하는 도구
LEFT_ALU = {
    (0, 1, 1, 0, 0): "+",
    (1, 1, 1, 0, 0): "*"
}

RIGHT_ALU = {
    (0, 1, 1, 0, 0): "-",
    (1, 1, 1, 0, 0): "/"
}

FIST = (0, 0, 0, 0, 0) # 주먹 상태 (아무것도 하지 않음)

# 양손 상태를 보고 연산자 결정하는 함수
def get_alu(left, right):
    # 손이 보이고 주먹이 아니라면 제스처중인 손
    left_active = left is not None and left != FIST
    right_active = right is not None and right != FIST

    # 양손이 모두 제스처 중이라면 애매하므로 무시
    if left_active and right_active:
        return None

    if left_active:
        if sum(left) == 1: # 손가락이 한개만 펴져있다면 
            return str(left.index(1)) # 펴진 손가락의 index를 반환
        if left in LEFT_ALU: # 손가락 상태가 LEFT_ALU에 정의되어 있다면 연산자 반환
            return LEFT_ALU[left] 
    if right_active: 
        if sum(right) == 1: 
            return str(right.index(1) + 5) # 오른손은 5~9번 손가락이므로 index에 5를 더해서 반환
        if right in RIGHT_ALU:
            return RIGHT_ALU[right]
    
    return None # 제스처가 없거나, 주먹이거나, 정의되지 않은 제스처라면 None 반환

# 계산기 함수
def calculate(expression):
    # 수식 문자열을 토큰으로 분리
    tokens = []
    num = "" # 숫자들
    for ch in expression:
        if ch.isdigit(): # 1받고 2를 받았다면 12로 만들기
            num += ch
        else: # 연산자를 받았을때
            if num == "": 
                return "Error" # 연속적인 연산자 입력 예외 처리
            tokens.append(int(num)) # 숫자 토큰 추가
            tokens.append(ch) # 연산자 토큰 추가
            num = "" # 숫자 초기화
    if num == "":
        return "Error" # 연산자로 끝나는 경우 예외 처리
    tokens.append(int(num)) # 마지막 숫자 토큰 추가 ex) 1+2*3이라면 tokens = [1, '+', 2, '*', 3]

    # 사칙연산의 우선순위는 곱셈 나눗셈부터
    stack = [tokens[0]] # 첫번째 숫자 토큰을 스택에 넣기
    for operator_idx in range(1, len(tokens), 2): # 연산자 토큰의 index는 1, 3, 5, ... 이므로 step=2
        operator, num = tokens[operator_idx], tokens[operator_idx + 1] # 1 * 2 -> tokens[1] = *, tokens[2] = 2
        if operator == "*":
            stack[-1] *= num # 스택의 마지막 숫자에 곱셈
        elif operator == "/":
            if num == 0: # 0으로 나누는 경우 예외 처리
                return "Error"
            stack[-1] /= num # 스택의 마지막 숫자에 나눗셈
        else: # +, - 연산자는 스택에 그대로 넣기 (후순위 계산)
            stack.append(operator)
            stack.append(num)

    # 덧셈, 뺄셈 왼쪽부터 차례대로 계산
    result = stack[0]
    for operator_idx in range(1, len(stack), 2):
        operator, num = stack[operator_idx], stack[operator_idx + 1]
        if operator == "+":
            result += num
        elif operator == "-":
            result -= num

    # 나눗셈 결과가 정수라면 소수점 없이, 소수라면 소숫점 2자리까지
    if result == int(result):
        return str(int(result))
    else:
        return f"{result:.2f}" # 소수점 2자리까지 표시


expression = "" # 수식 문자열
current_operator = None # 현재 연산자
hold_start = 0.0 # 그 연산자를 시작한 시각
ready = True # 중립을 거쳐서 다음 제스처를 기다리는 상태인지 여부
HOLD_TIME = 1.0 # 제스처를 HOLD_TIME초 이상 유지해야 연산자로 인식

# 포스트잇 검출 설정 (농구공 코드 기반)
lower_postit_lab = np.array([150, 100, 135], dtype=np.uint8) # 노란 포스트잇 LAB 하한
upper_postit_lab = np.array([255, 125, 200], dtype=np.uint8) # 노란 포스트잇 LAB 상한
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # 원형 커널 만들기
MIN_AREA = 3000 # 3000보다 작은건 포스트잇으로 보지 않음

mode = "INPUT" # INPUT: 수식 입력 모드, RESULT: 포스트잇에 결과 표시
result_text = "" 
visible_count = 0 # 포스트잇이 화면에 보이는 프레임 수
hidden_count = 0 # 포스트잇이 화면에 보이지 않는 프레임 수
SHOW_FRAMES = 10 # 포스트잇이 SHOW_FRAMES 이상 보이면 결과 표시
HIDE_FRAMES = 30 # 포스트잇이 HIDE_FRAMES 이상 안보이면 수식 입력 모드로 전환

# Kalman filter 상태[x, y, vx, vy], 측정[x, y]
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.diag([1e-2, 1e-2, 5e-2, 5e-2]).astype(np.float32) # 클수록 측정값 변화에 빠른 반응
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0 # 클수록 측정값 변화에 느린 반응
kalman.errorCovPost = np.eye(4, dtype=np.float32) # 초기 오차 공분산 행렬

found = False # 포스트잇이 화면에 보이는지 여부
x_bel, y_bel = 0.0, 0.0 # 칼만 필터 추정값
last_radius = 0 # 마지막 프레임에서 검출된 포스트잇 반지름

while True:
    ret, frame = cap.read() # 프레임 읽기, ret는 프레임을 제대로 읽었다면 True

    if not ret: # ret이 False라면 루프 탈출
        break
    if cv2.waitKey(1) & 0xFF == ord("q"): # q 키를 누르면 루프 탈출
        break

    # 프레임 높이와 너비
    h, w = frame.shape[:2] # H, W, C에서 H, W만 꺼내온다.

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    if platform.system() == "Windows": # Windows 노트북 용
        frame = cv2.flip(frame, 1) # 작성자의 환경(Windows)은 1로 처리 flip해줘야 화면이 거울모드가 됨.
    else: # NVIDIA Jetson Nano 용
        frame = cv2.flip(frame, 0) # NVIDIA Jetson Nano
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv는 BGR, mediapipe는 RGB이므로 색공간 변환 필요, rgb와 frame의 크기는 같음.

    # 포스트잇 검출(농구공 코드 기반) 가우시안 필터 사용
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)

    # BGR->LAB 색공간
    postit_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    # 채널 분할
    l, a, b = cv2.split(postit_lab)

    # CLAHE로 L 채널 강화한 enhanced image
    l_clahe_enhanced = clahe.apply(l)
    postit_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    # enhanced image로 binary mask 만들기
    mask_enhanced = cv2.inRange(postit_enhanced_lab, lower_postit_lab, upper_postit_lab)

    # 실험으로 검증한 Opening 3번
    mask_enhanced_opened3 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)
    final_postit_mask = mask_enhanced_opened3

    # 칸투어 검출
    contours, _ = cv2.findContours(
        final_postit_mask,       # binary mask
        cv2.RETR_EXTERNAL,       # 가장 바깥 외곽선만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 구간은 꼭짓점만 저장
    )
    
    postit_found = False
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea) # 가장 큰 contour 선택
        area = cv2.contourArea(largest_contour) # contour 면적 계산
        if area > MIN_AREA: # 면적이 최소 면적보다 크다면 포스트
            postit_found = True
            (x, y), radius = cv2.minEnclosingCircle(largest_contour) # 최소 외접원 계산
            center_circle = (int(x), int(y)) 
            radius = int(radius)

        
    cv2.imshow("Postit Mask", final_postit_mask) # 포스트잇 검출 결과 화면에 표시



    # 프레임 내 손 탐지
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) 
    result = hand_detector.detect(mp_image) 

    # 좌우 반전(거울모드)된 화면을 기준으로 왼손과 오른손 정보 변경
    labels = ["Left" if handedness[0].category_name == "Right" else "Right" for handedness in result.handedness]


    # 손마다 손가락 상태 튜플 만들기 (1 = 펴짐, 0 = 접힘)
    # result.hand_landmarks = [hand1, hand2, ...] (각 hand는 21개의 landmark 좌표를 가진 리스트)
    # labels는 ["Left", "Right"] 
    hand_states = {}
    for hand, label in zip(result.hand_landmarks, labels):
        state = []
        for point1_idx, point2_idx, point3_idx in finger_angle_points: # 손가락 관절 3개씩 묶어서 각도 계산
            angle = calculate_angle(
                hand[point1_idx],
                hand[point2_idx],
                hand[point3_idx],
            )
            state.append(1 if angle >= angle_threshold else 0) # 각도가 임계값 이상이면 1(펴짐), 아니면 0(접힘)으로 상태를 기록
            # 결국 angle 1개씩 받아서 state는 5개 값을 가진 리스트가 됨. (엄지, 검지, 중지, 약지, 소지 순서)
        hand_states[label] = tuple(state) # label을 key로, state를 value로 딕셔너리에 저장 
        # 예시: hand_states = {"Left": (1, 0, 1, 0, 1), "Right": (0, 1, 1, 1, 0)}


    # 손이 화면에 없다면 Key가 없으므로 None으로 처리
    if "Left" not in hand_states:
        hand_states["Left"] = None
        left = hand_states["Left"]
    else:
        left = hand_states["Left"]
    if "Right" not in hand_states:
        hand_states["Right"] = None
        right = hand_states["Right"]
    else:
        right = hand_states["Right"]

    # 화면 좌측 상단에 손 개수와 펼친 손가락 개수 표시
    cv2.putText(frame, f"Left: {left}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Right: {right}", (20,105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    operator = get_alu(left, right)

    # 모드 전환
    # 프레임 내 포스트잇이 보이면 visible_count 증가, 안보이면 hidden_count 증가
    if postit_found:
        visible_count += 1
        hidden_count = 0
    else: 
        hidden_count += 1
        visible_count = 0

    if mode == "INPUT" and visible_count >= SHOW_FRAMES: # 포스트잇이 SHOW_FRAMES 이상 보이면 결과 표시 모드로 전환
        result_text = calculate(expression) # 수식 계산
        mode = "RESULT"
    elif mode == "RESULT" and hidden_count >= HIDE_FRAMES: # 포스트잇이 HIDE_FRAMES 이상 안보이면 수식 입력 모드로 전환
        expression = "" # 수식 초기화
        result_text = ""
        mode = "INPUT"
        ready = False # 중립을 거쳐야 다음 제스처를 기다리는 상태로 전환

    # Kalman Filter 추적
    if hidden_count >= HIDE_FRAMES: # 포스트잇이 안보이면 추적 포기
        found = False
    
    if postit_found:
        last_radius = radius # 마지막 프레임에서 검출된 포스트잇 반지름 저장
        if not found:
            kalman.statePost = np.array([[center_circle[0]], [center_circle[1]], [0.0], [0.0]], dtype=np.float32) # 포스트잇이 처음 보이면 칼만 필터 상태 초기화
            found = True
        z = np.array([[center_circle[0]], [center_circle[1]]], dtype=np.float32) # 측정값
        kalman.predict() # 칼만 필터 예측
        state = kalman.correct(z) # 칼만 필터 보정
    elif found: 
        state = kalman.predict() # 포스트잇이 안보이면 칼만 필터 예측만 수행

    if found: 
        x_bel, y_bel = int(state[0, 0]), int(state[1, 0]) # 칼만 필터 추정값

    # 중립 = 양손 모두 주먹이거나 화면 밖
    neutral = (left is None or left == FIST) and (right is None or right == FIST)
    if mode == "RESULT": # 포스트잇이 보이면 결과 표시 모드
        current_operator = None # 연산자 초기화
    elif neutral: # 연산자 입력하고 중립거쳐야 다음 제스처를 기다리는 상태로 전환
        ready = True 
        current_operator = None 
    elif not ready or operator is None: # 연산자 계속 입력되는 것 방지
        current_operator = None
    elif operator != current_operator: # 연산자가 바뀌면 hold_start 초기화 + 시간 카운트
        current_operator = operator
        hold_start = time.time()
    elif time.time() - hold_start >= HOLD_TIME: # 연산자 유지 시간 체크
        expression += current_operator
        ready = False # 중립을 거쳐야 다음 제스처를 기다리는 상태로 전환
        current_operator = None # 연산자 초기화

    # 제스처 유지 시간 표시
    if current_operator is not None:
        hold_time = time.time() - hold_start
        cv2.putText(frame, f"Hold Time: {hold_time:.1f}s", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 입력된 수식 화면에 표시
    cv2.putText(frame, f"Expression: {expression}", (20,175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Mode: {mode}", (20,210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Result: {result_text}", (20,245), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
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
    # 포스트잇 위에 결과 표시
    if mode == "RESULT" and found:
        if postit_found: # 실제로 검출 되었을때만 외곽선 그림
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2) # 포스트잇 외곽선 그리기(초록색)

        # 첫 줄: 수식, 둘째 줄: 결과 -> 포스트잇 중신 기준으로 위아래에 배치
        lines = [expression, f"= {result_text}"]
        for line_idx, text in enumerate(lines):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = x_bel - text_size[0] // 2 # Kalman 추정 위치 기준 가운데 정렬
            text_y = y_bel - 10 + line_idx * 40 # 첫 줄 중심보다 위, 둘째 줄은 아래
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2) # 검정색 글자
    cv2.imshow("MediaPipe Hand Detection", frame)

hand_detector.close()
cap.release()
cv2.destroyAllWindows()