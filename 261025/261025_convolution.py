import numpy as np
import cupy as cp
import time
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    if padding > 0:
        image = cp.pad(image, padding, mode="constant")

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    output = cp.zeros((output_h, output_w), dtype=cp.float32)

    for kernel_y in range(kernel_h):
        for kernel_x in range(kernel_w):
            input_region = image[
                kernel_y:kernel_y + stride * output_h:stride,
                kernel_x:kernel_x + stride * output_w:stride
            ]

            output += input_region * kernel[kernel_y, kernel_x]

    output += bias

    return output

# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode="constant")

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w), dtype=np.float32)

    for output_y in range(output_h):
        for output_x in range(output_w):
            input_y = output_y * stride
            input_x = output_x * stride

            region = image[input_y:input_y + kernel_h, input_x:input_x + kernel_w]
            output[output_y, output_x] = np.sum(region * kernel) + bias

    return output


# 세로 경계 검출 커널 (x 방향 미분)
kernel_dx_np = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# 가로 경계 검출 커널 (y 방향 미분)
kernel_dy_np = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)


cpu_start = time.perf_counter()

output_dx_np = convolution2d(image=input_np, kernel=kernel_dx_np, padding=1)
output_dy_np = convolution2d(image=input_np, kernel=kernel_dy_np, padding=1)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

relu_dx_np = np.maximum(output_dx_np, 0)
relu_dy_np = np.maximum(output_dy_np, 0)

# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능

# 1. NumPy 배열을 CuPy 배열로 변환
upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
kernel_dx_cp = cp.asarray(kernel_dx_np)
kernel_dy_cp = cp.asarray(kernel_dy_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()
upload_time = upload_end - upload_start

# 2. 워밍업
warmup_image = cp.zeros((64, 64), dtype=cp.float32)

_ = convolution2d_cp(image=warmup_image, kernel=kernel_dx_cp, padding=1)

cp.cuda.Stream.null.synchronize()

# 3. GPU 처리
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(image=input_cp, kernel=kernel_dx_cp, padding=1)
output_dy_cp = convolution2d_cp(image=input_cp, kernel=kernel_dy_cp, padding=1)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()
gpu_time = gpu_end - gpu_start

# 4. ReLU 적용 후 CPU 메모리로 복사
download_start = time.perf_counter()

relu_dx_cp_np = cp.asnumpy(cp.maximum(output_dx_cp, 0))
relu_dy_cp_np = cp.asnumpy(cp.maximum(output_dy_cp, 0))

download_end = time.perf_counter()
download_time = download_end - download_start

# TODO: CPU/GPU 결과 비교 및 출력

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")
print()

print("========== Result Check ==========")
print("Vertical   (dx) same:", np.allclose(relu_dx_np, relu_dx_cp_np, atol=1e-5))
print("Horizontal (dy) same:", np.allclose(relu_dy_np, relu_dy_cp_np, atol=1e-5))

plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")