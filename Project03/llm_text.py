# 저장된 JSON 파일의 이벤트 물건으로 한국어 단어 따라 하기 → 문장 만들기 진행

import json
import os
import time

from llama_cpp import Llama

from speech import speech_to_text, text_to_speech


GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
JSON_PATH = "src/json/vision_data.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 60


# COCO 80개 클래스 → 한국어 단어
COCO_KO = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이",
    "airplane": "비행기", "bus": "버스", "train": "기차", "truck": "트럭",
    "boat": "배", "traffic light": "신호등", "fire hydrant": "소화전", "stop sign": "정지 표지판",
    "parking meter": "주차 요금기", "bench": "벤치", "bird": "새", "cat": "고양이",
    "dog": "강아지", "horse": "말", "sheep": "양", "cow": "소",
    "elephant": "코끼리", "bear": "곰", "zebra": "얼룩말", "giraffe": "기린",
    "backpack": "가방", "umbrella": "우산", "handbag": "핸드백", "tie": "넥타이",
    "suitcase": "여행 가방", "frisbee": "원반", "skis": "스키", "snowboard": "스노보드",
    "sports ball": "공", "kite": "연", "baseball bat": "야구 방망이", "baseball glove": "야구 글러브",
    "skateboard": "스케이트보드", "surfboard": "서핑보드", "tennis racket": "테니스 라켓", "bottle": "병",
    "wine glass": "와인잔", "cup": "컵", "fork": "포크", "knife": "칼",
    "spoon": "숟가락", "bowl": "그릇", "banana": "바나나", "apple": "사과",
    "sandwich": "샌드위치", "orange": "오렌지", "broccoli": "브로콜리", "carrot": "당근",
    "hot dog": "핫도그", "pizza": "피자", "donut": "도넛", "cake": "케이크",
    "chair": "의자", "couch": "소파", "potted plant": "화분", "bed": "침대",
    "dining table": "식탁", "toilet": "변기", "tv": "텔레비전", "laptop": "노트북",
    "mouse": "마우스", "remote": "리모컨", "keyboard": "키보드", "cell phone": "휴대폰",
    "microwave": "전자레인지", "oven": "오븐", "toaster": "토스터", "sink": "싱크대",
    "refrigerator": "냉장고", "book": "책", "clock": "시계", "vase": "꽃병",
    "scissors": "가위", "teddy bear": "곰 인형", "hair drier": "드라이기", "toothbrush": "칫솔",
}


llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


print("Vision JSON 입력 대기 중...")


# 파일이 마지막으로 수정된 시간 저장
if os.path.exists(JSON_PATH):
    last_modified = os.path.getmtime(JSON_PATH)
else:
    last_modified = 0


while True:
    # JSON 파일 존재 확인
    if os.path.exists(JSON_PATH):
        current_modified = os.path.getmtime(JSON_PATH)

        # JSON 파일이 수정되었는지 확인
        if current_modified != last_modified:
            last_modified = current_modified

            # 1. JSON 파일 → 이벤트 물건 이름을 한국어로 변환
            with open(JSON_PATH, "r", encoding="utf-8") as file:
                vision_data = json.load(file)

            word = COCO_KO[vision_data["event"]]

            print("\n[Vision Event]")
            print(f"{vision_data['event']} → {word}")


            # 2. 안내 문장 TTS
            guide_text = f"이것은 {word}입니다. 따라 해 보세요."

            print("\n[안내]")
            print(guide_text)

            text_to_speech(guide_text)


            # 3. 단어 따라 하기 STT
            user_text = input("Enter a word: ")

            print("\n[STT - 단어]")
            print(user_text)


            # 4. 단어 판정 후 문장 만들기 안내
            if word in user_text:
                sentence_text = f"잘했어요! 이번에는 {word}, 이 단어로 문장을 만들어 보세요."

                print("\n[안내]")
                print(sentence_text)

                text_to_speech(sentence_text)


                # 5. 문장 만들기 STT
                user_sentence = input("Enter a sentence: ")

                print("\n[STT - 문장]")
                print(user_sentence)


                # 6. 단어와 학습자 문장을 Gemma에 Context로 전달
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": """
                                        Instruction:
                                        한국어 학습자가 주어진 단어로 문장을 만들었다.
                                        문장이 자연스러운 한국어인지 판단하시오.

                                        Constraint:
                                        단어가 빠졌거나 문법이 틀리면 자연스러운 문장으로 고쳐 주시오.
                                        초급 학습자가 이해할 수 있는 쉬운 한국어로 답하시오.

                                        Output Format:
                                        첫째 줄 : 좋아요! 또는 아쉬워요.
                                        둘째 줄 : 좋아요!이면 칭찬 한 문장, 아쉬워요.이면 이렇게 말해 보세요. 고친 문장
                                    """
                        },
                        {
                            "role": "user",
                            "content": f"""
                                        Context:
                                        단어 : {word}
                                        학습자 문장 : {user_sentence}
                                    """
                        },
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=0.2,
                )

                answer = response["choices"][0]["message"]["content"]

                print("\n[Gemma]")
                print(answer)


                # 7. Gemma 피드백 TTS
                text_to_speech(answer)

            else:
                text_to_speech(f"아쉬워요. 다시 들어 보세요. {word}.")


            # 8. 대화 중 들어온 이벤트는 무시
            last_modified = os.path.getmtime(JSON_PATH)

            print("\nVision JSON 입력 대기 중...")


    # CPU를 불필요하게 계속 사용하지 않도록 잠시 대기
    time.sleep(0.1)
