import subprocess


MIC_DEVICE = "plughw:3,0"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5


print("말씀을 시작해 주세요.")


subprocess.run(
    [
        "pasuspender", "--",
        "arecord",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", str(RECORD_SECONDS),
        AUDIO_FILE,
    ],
    check=True,
)

result = subprocess.run(
    [
        WHISPER_PATH,
        "-m", WHISPER_MODEL,
        "-f", AUDIO_FILE,
        "-l", "ko",
        "--no-timestamps",
    ],
    text=True,
    capture_output=True,
    check=True,
)


text = result.stdout.strip()

print("STT 결과:")
print(text)
