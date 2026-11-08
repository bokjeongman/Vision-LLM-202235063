# 사물로 배우는 한국어 (Learn Korean with Objects)

202235063 송태규

카메라에 물건을 보여주면 그 물건의 한국어 이름을 알려주고, 따라 말하기와 문장 만들기를 음성으로 확인하는 한국어 말하기 학습 프로그램입니다.
YOLO로 화면 중앙의 물건을 감지하고, whisper.cpp로 학습자의 음성을 인식한 뒤, Gemma가 문장을 판정하고 Piper로 피드백을 음성 출력합니다.

외국인 유학생과 다문화 가정 학습자를 대상으로 합니다. COCO 80개 클래스(컵, 의자, 휴대폰, 사과 등)는 대부분 일상 사물이라 초급 어휘 목록으로 그대로 사용할 수 있습니다.

## 실행 환경

- NVIDIA Jetson Orin Nano 8GB (USB 카메라, USB 마이크, 스피커)
- GStreamer 파이프라인 (`/dev/video0`, 640x480, 30fps)
- Python 3.10.12

## 의존성

```bash
pip install ultralytics llama-cpp-python
```

- `llama-cpp-python`은 CUDA가 활성화된 빌드가 필요합니다.
- STT는 `whisper.cpp`를 CPU로 빌드하여 사용합니다. (`whisper.cpp/build-cpu/bin/whisper-cli`)
- TTS는 Piper를 별도 가상환경(`.piper_venv`)에 설치하여 사용합니다.
- 녹음과 재생에 `arecord`, `aplay`, `pasuspender`를 사용합니다.
- 다음 모델 파일을 각 경로에 두어야 합니다.
  - Gemma : `google_gemma-4-E2B-it-Q4_K_M.gguf` → `src/models/Gemma4/`
  - YOLO : `yolo11n_int8.engine` → `src/models/YOLO/`
  - Piper : `ko_KR-kss-medium.onnx` → `src/models/Piper/`
  - whisper : `ggml-base.bin` → `whisper.cpp/models/`
- 모델 파일, `whisper.cpp/`, `.piper_venv/`는 용량 문제로 Repository에 포함하지 않았습니다. 설치와 다운로드는 06 실습 노트북의 환경 구성과 동일합니다.
- `yolo11n_int8.engine`은 TensorRT 엔진으로, 실행할 Jetson 기기에서 직접 변환한 파일을 사용해야 합니다.

### whisper.cpp 빌드 (CPU)

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build-cpu
cmake --build build-cpu --target whisper-cli -j4 --config Release
```

### Piper 설치

```bash
python3 -m venv .piper_venv
source .piper_venv/bin/activate
pip install piper-tts
python3 -m piper.download_voices ko_KR-kss-medium --data-dir src/models/Piper
deactivate
```

## 폴더 구조

```
Project03/
├── vision.py
├── llm.py
├── speech.py
├── README.md
├── 발표자료.pdf
├── src/
│   ├── models/
│   │   ├── YOLO/yolo11n_int8.engine
│   │   ├── Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
│   │   └── Piper/ko_KR-kss-medium.onnx
│   ├── json/            # vision_data.json 저장 위치
│   └── audio/           # 녹음 파일, TTS 출력 파일 저장 위치
├── whisper.cpp/
│   └── models/ggml-base.bin
└── .piper_venv/
```

| 파일 | 역할 |
|---|---|
| `vision.py` | USB 카메라 + YOLO 탐지, 중앙 물건 이벤트 감지 후 JSON 저장 |
| `llm.py` | JSON 이벤트 감지, 한국어 단어 변환, 학습 대화 진행, Gemma 문장 판정 |
| `speech.py` | STT (arecord + whisper.cpp), TTS (Piper + aplay) |

## 실행 방법

처음 실행 전에 JSON과 음성 파일을 저장할 폴더를 만듭니다.

```bash
mkdir -p src/json src/audio
```

프로젝트 루트에서 터미널 두 개로 각각 실행합니다.

```bash
python vision.py
```

```bash
python llm.py
```

- `vision.py`와 `llm.py`는 별도 프로세스로 실행되며 `src/json/vision_data.json` 파일로 통신합니다.
- `q` 키를 누르면 `vision.py`가 종료됩니다.
- `l` 키를 누르면 현재 화면 중앙의 물건으로 강제 이벤트가 발생합니다. (시연 백업용)

## 사용 방법

### 1. 물건 보여주기

배우고 싶은 물건을 카메라 화면 **중앙**에 들고 약 1.5초(`HOLD_FRAMES`) 유지하면 학습이 시작됩니다.

### 2. 단어 따라 하기

시스템이 "이것은 휴대폰입니다. 따라 해 보세요."라고 말하면 녹음이 시작됩니다. 5초(`RECORD_SECONDS`) 안에 단어를 따라 말합니다.

- 인식 결과에 단어가 들어 있으면 다음 단계로 넘어갑니다.
- 단어가 없으면 "아쉬워요. 다시 들어 보세요. 휴대폰."이라고 정답을 다시 들려주고 끝납니다. 물건을 다시 보여주면 처음부터 다시 시작합니다.

### 3. 문장 만들기

시스템이 "잘했어요! 이번에는 휴대폰, 이 단어로 문장을 만들어 보세요."라고 말하면 해당 단어가 들어간 문장을 말합니다.

### 4. 피드백

Gemma가 문장이 자연스러운지 판단하여 음성으로 알려줍니다.

- 자연스러운 문장이면 "좋아요!"와 칭찬 한 문장
- 틀린 문장이면 "아쉬워요."와 고친 문장

### 예시: 사과

1. 사과를 화면 중앙에 1.5초 유지 → `[이벤트 감지] apple`
2. 🔊 "이것은 사과입니다. 따라 해 보세요."
3. 🎤 "사과"
4. 🔊 "잘했어요! 이번에는 사과, 이 단어로 문장을 만들어 보세요."
5. 🎤 "사과를 맛있어요."
6. 🔊 "아쉬워요. 이렇게 말해 보세요. 사과가 맛있어요."

## 동작 원리

### 중앙 물건 이벤트

- 매 프레임 YOLO 탐지 결과에서 BBox 중심점이 화면 중앙 영역(`CENTER_RATIO`, 가로·세로 50%) 안에 있는 물건만 후보로 선택합니다. 화면 가장자리의 배경 물건에는 반응하지 않습니다.
- 사람(`person`)은 후보에서 제외합니다.
- 후보가 여러 개면 BBox 면적이 가장 큰 물건, 즉 카메라 쪽으로 내민 물건을 선택합니다.
- 같은 물건이 `HOLD_FRAMES` 동안 연속으로 유지되면 이벤트가 발생합니다. 화면을 스쳐 지나가는 물건에는 반응하지 않습니다.
- 유지 프레임 수가 `HOLD_FRAMES`와 같아지는 순간 한 번만 이벤트가 발생하므로, 같은 물건을 계속 들고 있어도 반복되지 않습니다. 다시 학습하려면 물건을 내렸다가 다시 보여주면 됩니다.
- 이벤트가 발생하면 JSON의 `"event"`에 물건 이름을 저장합니다. 임시 파일에 먼저 쓴 뒤 `os.replace`로 교체하여 `llm.py`가 쓰다 만 파일을 읽지 않도록 합니다.

### 한국어 단어 변환

- COCO 80개 클래스를 한국어 단어로 매핑한 딕셔너리(`COCO_KO`)를 사용합니다. (예: `cell phone` → 휴대폰)
- 작은 LLM에 번역을 맡기면 결과가 매번 달라질 수 있어 고정 매핑을 사용합니다.
- 첫 안내 문장은 LLM 없이 템플릿으로 만들어 응답 속도를 줄였습니다. "입니다"는 받침 유무와 관계없이 자연스러워 조사 처리가 필요 없습니다.

### 단어 판정과 문장 판정

- 단어 따라 하기는 STT 결과에 정답 단어가 들어 있는지 문자열로 확인합니다. Gemma를 호출하지 않아 빠르게 판정합니다.
- 문장 만들기는 정답이 하나로 정해지지 않으므로 Gemma로 판정합니다. 단어와 학습자 문장을 Context로 전달하고, 첫째 줄에 `좋아요!/아쉬워요.`, 둘째 줄에 칭찬 또는 고친 문장을 출력하도록 합니다.
- 출력 형식 자체를 말로 읽을 수 있는 문장으로 정해 Gemma의 답을 별도 처리 없이 그대로 TTS에 전달합니다.
- 판정 결과가 일관되도록 `temperature`를 0.2로 낮추고, 짧은 피드백만 필요하므로 `MAX_TOKENS`를 60으로 제한했습니다.

### 대화 중 이벤트 처리

- `llm.py`는 JSON 파일의 수정 시간(`mtime`)을 감시하여 새 이벤트를 감지합니다.
- 대화가 끝나면 마지막 수정 시간을 현재 값으로 갱신하여, 대화 중에 들어온 이벤트는 무시합니다.

## 주요 파라미터

| 이름 | 값 | 설명 |
|---|---|---|
| `CENTER_RATIO` | 0.5 | 화면 중앙 영역 비율 |
| `HOLD_FRAMES` | 15 | 이벤트 판단 유지 프레임 수 |
| `conf` | 0.25 | YOLO 탐지 confidence 기준 |
| `RECORD_SECONDS` | 5 | 녹음 시간 |
| `CONTEXT_WINDOW` | 2048 | Gemma context 크기 |
| `MAX_TOKENS` | 60 | Gemma 최대 출력 토큰 |
| `temperature` | 0.2 | Gemma 출력 다양성 |

## 사용 모델

| 구분 | 모델 | 실행 |
|---|---|---|
| Vision | YOLO11n (`yolo11n_int8.engine`) | TensorRT INT8 |
| LLM | Gemma 4 E2B-it (`google_gemma-4-E2B-it-Q4_K_M.gguf`) | llama.cpp, GPU |
| STT | whisper base (`ggml-base.bin`) | whisper.cpp, CPU |
| TTS | Piper `ko_KR-kss-medium` | CPU |

## 과제 요구 기술 활용 정리

| 요구 기술 | 사용한 방법 | 사용 위치 |
|---|---|---|
| 객체 탐지 | YOLO11n TensorRT INT8 엔진으로 실시간 탐지 | `vision.py` |
| 이벤트 감지 | 화면 중앙 물건이 일정 프레임 유지되면 학습 시작 이벤트 발생 | `check_event`, `HOLD_FRAMES` |
| 프로세스 간 통신 | 탐지 결과와 이벤트를 JSON 파일로 저장, mtime 감시로 전달 | `vision_data.json`, `os.replace` |
| 음성 입력 (STT) | arecord 녹음 후 whisper.cpp로 한국어 인식 | `speech_to_text` |
| LLM | Gemma로 학습자 문장의 자연스러움 판정 및 교정 | `llm.create_chat_completion` |
| 음성 출력 (TTS) | Piper 한국어 모델로 안내와 피드백 음성 출력 | `text_to_speech` |

### Vision과 음성의 관계 표현

- **물건 → 학습 단어**: 카메라에 보인 물건이 그대로 이번에 배울 단어가 됩니다.
- **음성 → 판정**: 학습자가 말한 단어와 문장을 STT로 받아 정답 여부와 문장 교정에 사용합니다.
- **LLM → 피드백**: Gemma가 만든 피드백이 음성으로 학습자에게 돌아갑니다.

## 확장 방향

### 게이미피케이션

- 맞힌 단어 수와 연속 정답 횟수에 따라 점수와 레벨을 부여합니다.
- "컵을 찾아서 보여 주세요"처럼 시스템이 단어를 먼저 말하고 학습자가 물건을 찾아오는 미션 모드를 추가할 수 있습니다.

### 복습

- 현재는 학습 기록을 저장하지 않습니다. 틀린 단어와 문장을 파일로 저장하면 다음 학습 때 틀렸던 단어부터 다시 연습할 수 있습니다.

### 학습 어휘 확장

- 현재는 COCO 80개 클래스만 인식하므로, 교실, 기숙사, 마트, 병원, 주방처럼 특정 공간에서 쓰는 물건(칠판, 지우개, 장바구니, 체온계, 냄비 등)은 학습할 수 없습니다.
- 공간별 물건 이미지를 모아 YOLO를 추가 학습(fine-tuning)하고, 새 클래스를 `COCO_KO` 딕셔너리에 추가하면 됩니다. 이벤트 감지, 음성 입출력, Gemma 교정 흐름은 그대로 사용할 수 있습니다.
- 공간별로 모델을 따로 학습해 두면 "교실 모드", "마트 모드"처럼 상황에 맞는 단어 세트로 학습할 수 있습니다.

### 개인화

- 학습자 수준(초급, 중급)을 Gemma 프롬프트에 함께 전달해, 수준에 맞는 쉬운 표현 또는 조금 더 자연스러운 표현으로 교정할 수 있습니다.
- 학습자에 따라 자주 쓰는 물건이 다르므로, 유학생은 교실·기숙사 물건, 다문화 가정 아이는 음식·장난감처럼 생활 환경에 맞는 단어를 우선 학습하도록 할 수 있습니다. (COCO에 없는 물건은 위의 어휘 확장 필요)
- 복습 기록을 바탕으로 학습자가 자주 틀리는 조사나 문법이 들어간 문장을 더 연습하도록 유도할 수 있습니다.
- 말하는 속도가 느린 학습자를 위해 녹음 시간(`RECORD_SECONDS`)을 학습자별로 조절할 수 있습니다.

### 다양한 언어

- `COCO_KO` 딕셔너리, 안내 문장 템플릿, whisper 언어 설정(`-l ko`), Piper 음성 모델을 바꾸면 다른 언어 학습 프로그램으로 사용할 수 있습니다.
- 안내 문장만 학습자의 모국어로 들려주는 방식도 가능합니다.

### 로컬 LLM을 이용한 STT 보정

- 주변 소음이나 짧은 단어 때문에 whisper가 말을 잘못 인식하는 경우가 많습니다. (예: "휴대폰"이라고 말했는데 "핸드폰"으로 인식)
- 이런 경우는 공백 제거 같은 단순 규칙으로 처리하기 어렵기 때문에, STT 결과와 정답 단어를 로컬 LLM에 함께 전달해 같은 뜻의 다른 표기나 비슷한 단어로 잘못 인식된 경우를 보정하면 판정이 더 안정적입니다.
- 단, 보정이 지나치면 학습자의 실제 실수까지 통과시킬 수 있습니다.
  - 단어 단계: 학습자가 "가위"를 "가이"로 잘못 발음해도 "가위"로 보정되면 틀린 발음이 정답 처리됩니다.
  - 문장 단계: "사과를 맛있어요."가 "사과가 맛있어요."로 보정되면 문법 교정 기능이 의미가 없어집니다.
- 인식 오류와 학습자의 실수를 구분하기 어려우므로, 보정 범위를 좁게 제한하고 원래 STT 결과와 보정 결과를 함께 확인하는 방식이 필요합니다.

## 참고 사항

- 발음 판정 기준은 "whisper가 알아들을 수 있는가"입니다. 실제 발음 평가 모델은 아닙니다.
- 주변 소음이 크면 인식률이 떨어지므로, 마이크 가까이에서 또박또박 말해야 안정적으로 동작합니다.
- "컵", "책"처럼 한 글자 단어는 whisper가 잘못 인식할 수 있어 두 글자 이상 단어가 더 안정적입니다.
- STT 결과에 정답 단어가 정확히 들어 있어야 통과합니다. (예: 휴대폰을 "핸드폰"으로 인식하면 오답)
- 한 번에 한 물건만 학습할 수 있고, 대화 중에 보여준 물건은 무시됩니다.
- 화면 중앙에 물건이 있어도 YOLO가 한 프레임이라도 놓치면 유지 프레임 수가 다시 측정됩니다.
- Gemma가 출력 형식을 지키지 않으면 피드백이 어색하게 읽힐 수 있습니다.