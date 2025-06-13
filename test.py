import json

from openai import OpenAI

client = OpenAI()
audio_file = open("projects/test/ea48283a31baa560/audio.wav", "rb")

transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["segment"]
)

with open("transcription.json", "w", encoding="utf-8") as f:
    f.write(transcription.to_json())

print(transcription)