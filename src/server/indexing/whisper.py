import os
import subprocess


def audio_voice_analysis(audio_path: str):


if __name__ == "__main__":
    input_video = "data/20230106_161018.mp4"
    cache_path = cache_audio(input_video)

    # 2) Whisper 모델 로드
    asr = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device_map="auto",
        chunk_length_s=30,
        stride_length_s=(5, 5),
        batch_size=1,
        return_timestamps="sentence",   # 단어 단위 타임스탬프
        return_language=True        # 언어 정보
    )

    # 1) Torch에서 GPU 사용 가능 여부
    print("CUDA available:", torch.cuda.is_available())

    # 2) 파이프라인 내부 모델이 올라가 있는 디바이스
    device = asr.model.device
    print("Pipeline model device:", device)

    # 3) 전사 및 타임스탬프, 언어 정보 요청
    result = asr(cache_path)

    # 4) 전체 텍스트 출력
    print("=== Full transcription ===")
    print(result["text"])

    # 5) 단어별 타임스탬프 출력
    print("\n=== Word-level timestamps ===")
    for chunk in result["chunks"]:
        text = chunk["text"].strip()
        start, end = chunk["timestamp"]

        lang = chunk.get("language", "undetected")
        print(f"{start:.2f}s → {end:.2f}s | [{lang}] {text}")
