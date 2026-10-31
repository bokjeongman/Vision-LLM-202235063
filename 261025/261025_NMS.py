from random import randint, gauss
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
import cv2

# draw_bbox: 이미지에 bounding box를 그리는 함수
def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# random_bbox: 이미지에서 랜덤한 bounding box를 생성하는 함수
def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

# random_conf: 랜덤한 confidence score를 생성하는 함수
def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의
# convert_bbox_coord: BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수
def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)

# calculate_iou: 두 BBox의 Intersection over Union (IoU)를 계산하는 함수
def calculate_iou(bbox_a, bbox_b):
    xa1, ya1, xa2, ya2 = bbox_a
    xb1, yb1, xb2, yb2 = bbox_b

    # 교집합 사각형 좌표
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    # 교집합 넓이 (안 겹치면 0)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    # 합집합 넓이
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - intersection_area
    # TODO: ZeroDivisionError 주의
    if union_area == 0:
        return 0
    return intersection_area / union_area

hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []
# N개의 랜덤한 BBox와 Confidence Score 생성
for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

# 랜덤한 BBox와 Confidence Score를 이미지에 그리기
for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거
# NMS 과정 (2~6번):
# 1. 신뢰도가 너무 낮은 BBox 제거 (Confidence Thresholding)
# 2. Confidence가 가장 높은 BBox 선택
# 3. 선택한 BBox와 다른 박스들의 IoU 계산
# 4. IoU가 임계값보다 높은 박스 제거 (IoU Thresholding)
# 5. 다음으로 Confidence가 가장 높은 BBox 선택
# 6. 3, 4, 5번 반복


def prune_low_conf(boxes, scores, conf_threshold=0.5):
    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores): # boxes와 scores를 동시에 순회
        if score >= conf_threshold:
            pruned_boxes.append(box) # Confidence Score가 임계값 이상인 BBox만 유지
            pruned_scores.append(score) # Confidence Score가 임계값 이상인 BBox의 Confidence Score만 유지

    return pruned_boxes, pruned_scores

    # """
    # Confidence Thresholding 적용

    # Parameters
    # ----------
    # boxes : list
    #     Bounding Box 목록
    #     각 박스는 (x1, y1, x2, y2) 형식

    # scores : list
    #     각 Bounding Box의 Confidence Score

    # conf_threshold : float
    #     Bounding Box를 유지하기 위한 최소 Confidence Score

    # Returns
    # -------
    # pruned_boxes : list
    #     Confidence Score가 임계값 이상인 Bounding Box 목록

    # pruned_scores : list
    #     Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    # """



# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환
# NMS 과정 (2~6번):
# 1. 신뢰도가 너무 낮은 BBox 제거 (Confidence Thresholding)
# 2. Confidence가 가장 높은 BBox 선택
# 3. 선택한 BBox와 다른 박스들의 IoU 계산
# 4. IoU가 임계값보다 높은 박스 제거 (IoU Thresholding)
# 5. 다음으로 Confidence가 가장 높은 BBox 선택
# 6. 3, 4, 5번 반복
def apply_nms(boxes, scores, iou_threshold=0.5):
    # score 내림차순 인덱스
    order = list(np.argsort(scores)[::-1])

    keep = []
    while order:
        # 2, 5번: 남은 박스 중 score가 가장 높은 박스 선택
        best = order.pop(0)
        keep.append(best)

        # 3, 4번: 선택한 박스와 IoU가 임계값 이하인 박스만 남김
        order = [
            i for i in order
            if calculate_iou(boxes[best], boxes[i]) <= iou_threshold
        ]
        # 6번: order가 빌 때까지 반복

    nms_boxes = [boxes[i] for i in keep]
    nms_scores = [scores[i] for i in keep]

    return nms_boxes, nms_scores

    # """
    # NMS 적용

    # Parameters
    # ----------
    # boxes : list
    #     Bounding Box 목록
    #     각 박스는 (x1, y1, x2, y2) 형식

    # scores : list
    #     각 Bounding Box의 Confidence Score

    # iou_threshold : float
    #     중복 Bounding Box를 제거할 IoU 기준값

    # Returns
    # -------
    # nms_boxes : list
    #     NMS 적용 후 유지된 Bounding Box 목록

    # nms_scores : list
    #     NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    # """


print(len(boxes), len(scores))

# Confidence Thresholding 적용
new_boxes, new_scores = prune_low_conf(boxes, scores)
# NMS 적용
new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

hubble = skimage.data.hubble_deep_field()
# 랜덤한 색상으로 BBox를 이미지에 그리기
for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{new_scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """
    OpenCV의 cv2.dnn.NMSBoxes()를 이용해 NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록
        각 박스는 기존과 동일한 (x1, y1, x2, y2) 형식

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score
    """

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes: # (x1, y1, x2, y2) → (x, y, w, h) 변환
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    # OpenCV NMS 적용하여 BBox 제거 후 남은 BBox의 인덱스
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    # 1차원 리스트로 변환
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    # 원래 xyxy 형식의 박스에서 선택
    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores

print(len(boxes), len(scores))

cv2_new_boxes, cv2_new_scores = apply_nms_cv2(boxes, scores, iou_threshold=0.2) # OpenCV NMS 적용

print(len(cv2_new_boxes), len(cv2_new_scores))

hubble_cv2_nms = skimage.data.hubble_deep_field()

for i, box in enumerate(cv2_new_boxes): # NMS 결과 BBox를 이미지에 그리기
    draw_bbox(
        hubble_cv2_nms,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{cv2_new_scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble_cv2_nms)
plt.axis("off");

plt.figure(figsize=(16,10))

plt.subplot(1,2,1), plt.imshow(hubble), plt.title("Custom NMS Result")
plt.subplot(1,2,2), plt.imshow(hubble_cv2_nms), plt.title("CV2 NMS Result")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()