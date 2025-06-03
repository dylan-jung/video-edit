import argparse
import os
import time
from pathlib import Path

import google.generativeai as genai


def ask_gemini(video_path: str, prompt: str, model: str = "gemini-2.0-flash") -> str:
    """
    Send *video_path* together with *prompt* to Gemini and return the response text.
    Uses File API to upload video instead of base64 encoding.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError(
            "환경변수 GOOGLE_API_KEY 가 설정되어 있지 않습니다. "
            "https://ai.google.dev/ 에서 API 키를 발급받아 설정하세요."
        )

    # Configure the API key
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Upload video file using File API
    print(f"비디오 파일 업로드 중: {video_path}")
    video_file = genai.upload_file(path=video_path)
    print(f"업로드 완료. 파일 URI: {video_file.uri}")

    # Wait for the file to be processed
    while video_file.state.name == "PROCESSING":
        print("파일 처리 중...")
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"비디오 파일 처리 실패: {video_file.state.name}")

    # Initialize the model
    model_instance = genai.GenerativeModel(model)

    # Generate content with the uploaded video and prompt
    response = model_instance.generate_content([
        prompt + f"\n이 동영상은 전체 동영상을 여러 개의 청크로 나누었고, 2번째 청크입니다.",
        "이 동영상을 분석해줘",
        video_file
    ])

    # Clean up: delete the uploaded file
    genai.delete_file(video_file.name)
    print("업로드된 파일 삭제 완료")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini 2.0-Flash video+text demo (google.generativeai)"
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 file (≤ 2 min)")
    parser.add_argument("prompt", help="Text prompt to send with the video")
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro-preview-05-06",
        help="Gemini model ID (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(args.prompt, "r") as f:
        prompt = f.read()

    reply = ask_gemini(str(args.video), prompt, model=args.model)
    print("\n=== Gemini 답변 ===\n")
    print(reply)


if __name__ == "__main__":
    main()