# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    # 초록 LAB lower/upper range
    green_lower = np.array([30, 60, 90], dtype=np.uint8) # lower bound for basketball in LAB color space
    green_upper = np.array([230, 115, 180], dtype=np.uint8) # upper bound for basketball in LAB color space

    # TODO: 각 프레임 이미지를 전처리
    # TODO: Color Segmentation으로 Binary Mask 생성
    # TODO: Morphological Operation을 적용하여 객체 영역 추출
    # TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay

   

    image_per_frame = frame 

    # Convert BGR to LAB
    ipr_lab = cv2.cvtColor(image_per_frame, cv2.COLOR_RGB2LAB)
    '''
    # Split each channel
    l, a, b = cv2.split(ipr_lab)

    # Generate CLAHE (Set clipLimit 3.0, tileGridSize 8x8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Generate enhanced image by applying CLAHE to the L channel
    l_clahe_enhanced = clahe.apply(l)
    ipr_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b)) # Merge enhanced L, a, b channels to get the enhanced LAB image
    '''

    # Generate binary mask by enhanced image
    # mask_enhanced = cv2.inRange(ipr_enhanced_lab, green_lower, green_upper)
    mask_not_enhanced = cv2.inRange(ipr_lab, green_lower, green_upper)

    # Convert enhanced LAB image to RGB for visualization
    # ipr_enhanced_rgb = cv2.cvtColor(ipr_enhanced_lab, cv2.COLOR_LAB2RGB)
    ipr_not_enhanced_rgb = cv2.cvtColor(ipr_lab, cv2.COLOR_LAB2RGB)

    # Create a circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

    # Apply morphological operation and generate final binary mask 
    # Opening operation 3 times to remove noise and small objects
    # mask_enhanced_opened3 = cv2.morphologyEx(mask_not_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)

    # final_ipr_mask = mask_enhanced_opened3 # final mask
    final_ipr_mask = mask_not_enhanced 

    # final_img = ipr_enhanced_rgb.copy() # final image
    final_not_enhanced_img = ipr_not_enhanced_rgb.copy()
    contours, _ = cv2.findContours(
        final_ipr_mask,        # binary mask
        cv2.RETR_EXTERNAL,       # detect only the outer edges of the object with holes
        cv2.CHAIN_APPROX_SIMPLE  # store only the corner coordinates instead of all pixel coordinates along a straight path
    )
    found = False
    while True: 
        frame = cap.read()[1]
        if len(contours) > 0:
            # The largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center_circle = (int(x), int(y))
            radius = int(radius)
            # Moments를 통해 무게중심 (Centroid) 산출
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = center_circle
            
            centroid = (cX, cY)
            # 최소 외접원 반지름
            #radius = ...
            # 무게중심
            #center = ...
            # 검출된 객체에 파란 원 overlay
            cv2.circle(frame, centroid, int(radius), (0, 0, 255), 3)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            if not found: 
                found = True

            # 최초 검출 이후 Kalman Filter 적용
        # 측정값 사용 (visible) -> Prediction & Update
        if found and (len(contours) > 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(centroid)]))
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]

        # 측정값 미사용 (occluded) -> Prediction
        elif found and (len(contours) <= 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]

        # 예측한 객체 위치에 노란 원 overlay
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (255, 255, 0), 3)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (255, 255, 0), -1)
        # Generate the largest contour only filled binary mask
        final_mask = np.zeros_like(final_ipr_mask)
        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # The largest contour only filled binary mask to overlay with a semi-transparent green color
        mask_color = np.zeros_like(final_not_enhanced_img)
        mask_color[:, :, 1] = final_mask
        alpha = 0.35
        final_img = cv2.addWeighted(final_not_enhanced_img, 1.0, mask_color, alpha, 0)

        # Draw the largest contour (green outline)
        # cv2.drawContours(final_not_enhanced_img, [largest_contour], -1, (0,255,0), 2)
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.imshow("VideoCapture with GStreamer", cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2))

cap.release()

