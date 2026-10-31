# 261025 과제

## 제출 파일

| 파일 | 내용 |
|---|---|
| `261025_afteimageEffect.py` | 잔상 효과 NumPy(CPU) / CuPy(GPU) 구현 및 처리 시간 비교 |
| `261025_convolution.py` | 2D Convolution NumPy(CPU) / CuPy(GPU) 구현 및 처리 시간 비교 |
| `261025_DL_MNIST.py` | MNIST CNN 학습(CPU), Feature Map 시각화, 모델 저장 및 불러오기 |
| `261025_MNIST_main_gpu.py` | MNIST CNN `main_cpu()` / `main_gpu()` 학습 시간 비교 |
| `261025_CIFAR10_training.py` | CIFAR-10 CNN 학습(GPU), 모델 저장 및 불러오기 |
| `CIFAR10_CNN.pth` | `261025_CIFAR10_training.py`로 학습 후 저장한 모델 |
| `261025_IoU.py` | 사과 이미지 3장에서 두 BBox의 IoU 계산 |
| `261025_NMS.py` | Confidence Thresholding, NMS 구현 및 OpenCV NMS와 비교 |

## 폴더 구조

```
.
├── 261025/
│   ├── README.md
│   ├── 261025_afteimageEffect.py
│   ├── 261025_convolution.py
│   ├── 261025_DL_MNIST.py
│   ├── 261025_MNIST_main_gpu.py
│   ├── 261025_CIFAR10_training.py
│   ├── CIFAR10_CNN.pth
│   ├── 261025_IoU.py
│   └── 261025_NMS.py
└── src/
    ├── images/      # seagull.jpg, two_apples1.png, two_apples2.png, two_apples3.png
    ├── datasets/    # MNIST
    └── models/
        └── MNIST/   
```

## 의존성

- PyTorch, torchvision
- NumPy
- CuPy
- OpenCV
- Matplotlib
- scikit-image

```bash
pip install torch torchvision numpy opencv-python matplotlib scikit-image
pip install cupy-cuda12x   # 설치된 CUDA 버전에 맞는 CuPy 설치
```

## 실행 방법

최상위 폴더(`261025`와 `src`가 있는 위치)에서 실행합니다.

```bash
python 261025/261025_afteimageEffect.py
python 261025/261025_convolution.py
python 261025/261025_DL_MNIST.py
python 261025/261025_MNIST_main_gpu.py
python 261025/261025_CIFAR10_training.py
python 261025/261025_IoU.py
python 261025/261025_NMS.py
```

## 파일별 설명

### `261025_afteimageEffect.py`

`src/images/seagull.jpg`를 흑백 1280×720으로 불러와 잔상 효과를 적용합니다. 픽셀을 가로로 0~31칸 이동시키며 가중치를 곱해 누적하고, 가중치는 이동 거리가 멀수록 작아지며 합이 1이 되도록 정규화합니다.

- `motion_trail_np`: 출력 픽셀을 Python 반복문으로 하나씩 계산
- `motion_trail_cp`: 이동 거리마다 이미지 전체를 slicing해 GPU에서 계산

CPU 처리 시간, CPU→GPU 전송 시간, GPU 처리 시간, GPU→CPU 전송 시간과 속도 향상 배율(연산만 / 전송 포함)을 출력합니다.

### `261025_convolution.py`

`src/images/seagull.jpg`를 흑백 1280×720으로 불러와 세로 경계 검출 커널(x 방향 미분)과 가로 경계 검출 커널(y 방향 미분)로 2D Convolution(padding=1)과 ReLU를 적용합니다.

- `convolution2d`: 출력 위치마다 반복하며 NumPy로 계산
- `convolution2d_cp`: 커널 위치마다 출력 배열 전체를 CuPy로 계산

처리 시간과 속도 향상 배율을 출력하고, CPU/GPU 결과가 같은지 `np.allclose`로 확인한 뒤 입력 이미지와 결과를 시각화합니다.

### `261025_DL_MNIST.py`

MNIST CNN을 CPU에서 3 Epoch 학습합니다.

- 모델 구조: `Conv(1→8) → ReLU → Pool → Conv(8→16) → ReLU → Pool → FC(784→64) → FC(64→10)`
- 학습 전 Feature Map(conv1, relu1, pool1, relu2, pool2)과 Conv1 필터 시각화
- 학습 후 test 이미지 예측 결과, Loss / Accuracy 그래프
- Conv1 Feature Map 8개, Conv2 Feature Map 16개, 6개 Layer별 Feature Map 4개씩 비교 (`viridis`)
- 모델을 `src/models/MNIST/MNIST_CNN.pth`에 저장 후 불러오기

### `261025_MNIST_main_gpu.py`

같은 MNIST CNN을 `main_cpu()`와 `main_gpu()`로 각각 3 Epoch 학습하고, Epoch별 결과와 전체 학습 시간, 시간 차이, GPU 속도 향상 배율을 출력합니다. GPU 학습 시간은 `torch.cuda.synchronize()`로 GPU 연산이 끝난 시점까지 측정합니다.

### `261025_CIFAR10_training.py`

CIFAR-10 CNN을 GPU에서 5 Epoch 학습합니다.

- 학습 데이터: RandomCrop(padding=4), RandomHorizontalFlip, Normalize
- test 데이터: Normalize
- 모델 구조: `ConvBlock(3→64) → ConvBlock(64→128) → ConvBlock(128→256) → Global Average Pooling → Dropout(0.3) → FC(256→10)`
  - ConvBlock: `Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool`
- Optimizer: AdamW (lr=0.001)
- 학습 후 test 이미지 예측 결과, Loss / Accuracy 그래프
- 모델을 `261025/CIFAR10_CNN.pth`에 저장 후 불러와서 예측 확인

`CIFAR10_CNN.pth`에는 `model_state_dict`, `optimizer_state_dict`, 마지막 test accuracy, Epoch 수, 클래스 이름, 정규화 평균/표준편차가 저장되어 있습니다.

### `261025_IoU.py`

`src/images/two_apples1~3.png`에서 HSV 색 범위로 빨간 사과와 초록 사과의 가장 큰 Contour를 추출하고 Bounding Box를 구합니다. 두 BBox의 IoU를 이미지별로 계산해 비교합니다.

```
IoU = 교집합 넓이 / 합집합 넓이
```

### `261025_NMS.py`

`skimage.data.hubble_deep_field()` 이미지에 랜덤 BBox 20개와 Confidence Score를 생성합니다.

- `prune_low_conf`: Confidence Score가 0.5 미만인 BBox 제거
- `apply_nms`: Confidence가 높은 순서로 BBox를 선택하고, 선택한 BBox와 IoU가 0.2보다 큰 BBox를 제거하는 과정을 반복
- `apply_nms_cv2`: `cv2.dnn.NMSBoxes()`로 같은 조건의 NMS 적용

직접 구현한 NMS 결과와 OpenCV NMS 결과를 나란히 시각화합니다.

## 참고 사항

- `261025_afteimageEffect.py`, `261025_convolution.py`, `261025_MNIST_main_gpu.py`, `261025_CIFAR10_training.py`는 CUDA GPU가 필요합니다.
