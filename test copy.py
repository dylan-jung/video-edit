import json

with open("transcription.json", "r", encoding="utf-8") as f:
    transcription = json.load(f)

with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(transcription)

print(transcription)