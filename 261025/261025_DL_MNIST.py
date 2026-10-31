import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

# 데이터를 불러올 때 이미지 전처리 방식 정의
# PIL 이미지 또는 NumPy 이미지를 PyTorch Tensor로 변환
# 일반적인 8비트 이미지의 픽셀값을 0~1 범위의 실수로 변환
transform = transforms.ToTensor()

# 학습 데이터 불러오기 (최초 다운로드)
train_dataset = datasets.MNIST(
    root="src/datasets",
    train=True,
    transform=transform,
    download=True,
)

# 테스트 데이터 불러오기 (최초 다운로드)
test_dataset = datasets.MNIST(
    root="src/datasets",
    train=False,
    transform=transform,
    download=True,
)

print(
    "Train data:",
    len(train_dataset)
)

print(
    "Test data:",
    len(test_dataset)
)

image, label = train_dataset[23964]

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Label:", label)
print("Minimum:", image.min().item())
print("Maximum:", image.max().item())
print(train_dataset)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d( # Max Pooling
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7, # input 784
            64,         # output 64
        )

        self.fc2 = nn.Linear(
            64,         # input 64
            10,         # output 10
        )

    def forward(self, x, return_features=False): # network modeling
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

        return logits

device = torch.device("cpu")

torch.manual_seed(seed)

model = SimpleCNN()
model = model.to(device)

print("Device:", device)
print(model)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_parameter_count)

dummy_input = torch.randn(1, 1, 28, 28)
dummy_output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", dummy_output.shape) # 1 image to 10 categories predict the raw logut and then apply softmax-> it could be the probabilties

sample_images, sample_labels = next(
    iter(train_loader)
)

sample_images = sample_images.to(device)

with torch.no_grad():
    logits, features = model(
        sample_images,
        return_features=True,
    )

print("Input:", sample_images.shape)

for name, feature in features.items():
    print(f"{name}:", feature.shape)

print("Logits:", logits.shape)

def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray"):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()

    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
        plt.suptitle(title)

image, label = test_dataset[5678]

input_batch = image.unsqueeze(0).to(device)

print("Single image:", image.shape)
print("Batch image:", input_batch.shape)

model.eval()

with torch.no_grad():
    logits, features = model(
        input_batch,
        return_features=True,
    )

plt.figure(figsize=(4,4))

plt.imshow(image.squeeze(0), cmap="gray"), plt.title(f"Input label: {label}")
plt.axis("off")

show_feature_maps(
    features["conv1"],
    title="Conv1 Feature Maps",
    max_maps=8,
    cmap="viridis" # add cmap="viridis" to change gray to viridis
)
for i in range(8):
    conv = features["conv1"][0, i]

    print(
        f"Channel {i}: "
        f"Conv min={conv.min().item():.6f}, "
        f"Conv max={conv.max().item():.6f}, "
        )
    
show_feature_maps(
    features["relu1"],
    title="Conv1 + ReLU Feature Maps",
    max_maps=8,
)

show_feature_maps(
    features["pool1"],
    title="Pool1 Feature Maps",
    max_maps=8,
)

show_feature_maps(
    features["relu2"],
    title="Conv2 + ReLU Feature Maps",
    max_maps=12,
    cmap = "viridis"
)

show_feature_maps(
    features["pool2"],
    title="Pool2 Feature Maps",
    max_maps=12,
)

conv1_weights = model.conv1.weight.detach().cpu()
print(conv1_weights.shape) # 8features, Grayscale, H, W ->8, 1, 3, 3

plt.figure(figsize=(12,5))

for filter_index in range(conv1_weights.shape[0]):
    plt.subplot(2, 4, filter_index + 1)
    plt.imshow(conv1_weights[filter_index, 0], cmap="gray")
    plt.title(f"Filter {filter_index}")
    plt.axis("off")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 모델을 학습 모드(training mode)로 변경
    model.train()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 학습 데이터를 DataLoader(train_loader)의 Batch 단위로 불러와 학습 반복
    for images, labels in data_loader:
        # CPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화 (이전 Batch에서 계산한 Gradient를 제거)
        optimizer.zero_grad()

        # 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)
        # 역전파를 통한 Gradient 계산
        loss.backward()
        # Gradient를 사용하여 파라미터 수정
        optimizer.step()

        # Batch 한 개의 이미지 개수
        batch_size = images.size(0)

        # 결과 누적
        total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
        predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
        correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
        sample_count += batch_size                             # 전체 샘플 수 누적

    # 평균 Loss 및 Accuracy 계산
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    # 모델을 평가 모드(evaluation mode)로 변경
    model.eval()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # Gradient 계산을 비활성화 (파라미터 수정 비활성화)
    with torch.no_grad():
        # 평가 데이터를 DataLoader(test_loader)의 Batch 단위로 불러와 평가 반복
        for images, labels in data_loader:
            # CPU로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
            logits = model(images)

            # Loss 계산 (역전파는 수행하지 않음)
            loss = criterion(logits, labels)

            # Batch 한 개의 이미지 개수
            batch_size = images.size(0)

            # 결과 누적
            total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
            predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
            correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
            sample_count += batch_size                             # 전체 샘플 수 누적

        # 평균 Loss 및 Accuracy 계산
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy
    
# 총 Epoch 정의 (전체 학습 데이터를 몇 번 반복해서 볼 것인지)
epochs = 3

# 학습 기록 저장 공간 준비
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작
training_start_time = time.perf_counter()

for epoch in range(epochs):
    # Epoch 시작 시간 저장
    epoch_start_time = time.perf_counter()

    # 모델 학습
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    # 모델 평가
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # 결과 저장 (학습 & 평가)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    # Epoch 소요 시간 계산
    epoch_time = time.perf_counter() - epoch_start_time

    # 현재 Epoch의 학습/평가 결과 출력
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# 전체 학습 시간 계산 및 출력
total_training_time = time.perf_counter() - training_start_time
print(f"Total CPU training time: {total_training_time:.2f}s")

# 학습된 모델로 이미지 하나 테스트
image, true_label = test_dataset[0]          # 테스트할 이미지(image)와 정답(label)
input_batch = image.unsqueeze(0).to(device)  # Batch 차원 추가(unsqueeze)하여 입력 Batch로 저장 (이미지 1장)

# 모델을 평가 모드(evaluation mode)로 변경
model.eval()

# Gradient 계산을 비활성화 (파라미터 수정 비활성화)
with torch.no_grad():
    logits = model(input_batch)                           # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
    probabilities = torch.softmax(logits, dim=1)          # 원시 점수에 softmax 함수 적용 (클래스별 확률 계산)
    predicted_label = probabilities.argmax(dim=1).item()  # 정답 예측

# 결과 출력 (정답, 예측, confidence)
print("True label:", true_label)
print("Predicted label:", predicted_label)
print("Confidence:", probabilities[0, predicted_label].item())

# 결과 이미지 출력
plt.figure(figsize=(4,4))
plt.imshow(image.squeeze(0), cmap="gray")
plt.title(f"True: {true_label}, " f"Prediction: {predicted_label}")
plt.axis("off")
plt.show()

plt.figure(figsize=(15,10))

num_images = 40

model.eval()

with torch.no_grad():
    for i in range(num_images):
        image, true_label = test_dataset[i]
        input_batch = image.unsqueeze(0).to(device)

        logits = model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()

        plt.subplot(4, 10, i+1)
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"Conf: {confidence*100:.1f}%"
        )
        plt.axis("off")

plt.tight_layout()
plt.show()

# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8,5))

# TODO: train_loss_history 그래프 생성
plt.plot(train_loss_history)
# TODO: test_loss_history 그래프 생성
plt.plot(test_loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.grid()
plt.legend()
plt.show()

# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8,5))



# TODO: train_accuracy_history 그래프 생성 (백분율)
# TODO: test_accuracy_history 그래프 생성 (백분율)

plt.plot(np.array(train_accuracy_history) * 100)
plt.plot(np.array(test_accuracy_history) * 100)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()

# TODO: CNN 각 층의 Feature Map 시각화
# TODO: cmap="viridis" 사용


# TODO: 이미지 선택
image, true_label = test_dataset[5678]

# TODO: 역전파(파라미터 수정) 없이 순전파 수행
model.eval()
with torch.no_grad():
    input_batch = image.unsqueeze(0).to(device)
    logits, features = model(
        input_batch,
        return_features=True,
    )

# TODO: Conv1 Feature Maps 8개 시각화
show_feature_maps(
    features["conv1"],
    title="Conv1 Feature Maps",
    max_maps=8,
    cmap="viridis"
)

# TODO: Conv2 Feature Maps 16개 시각화
show_feature_maps(
    features["conv2"],
    title="Conv2 Feature Maps",
    max_maps=16,
    cmap="viridis"
)

# TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
layer_names = [
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2"
]

# 6개 Layer별 대표 Feature Map 4개씩 비교
plt.figure(figsize=(10, 14))

for row, name in enumerate(layer_names):
    feature_maps = features[name][0].cpu()  # [Channel, H, W]

    for col in range(4):
        plt.subplot(len(layer_names), 4, row * 4 + col + 1)
        plt.imshow(feature_maps[col], cmap="viridis")
        plt.title(f"{name} - Ch {col}")
        plt.axis("off")

plt.tight_layout()
plt.show()

torch.save(
    model.state_dict(),
    "src/models/MNIST/MNIST_CNN.pth",
)

loaded_model = SimpleCNN()

state_dict = torch.load(
    "src/models/MNIST/MNIST_CNN.pth",
    map_location="cpu",
    weights_only=True,
)

loaded_model.load_state_dict(state_dict)
loaded_model.to(device)
loaded_model.eval()



