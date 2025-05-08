import os

import replicate


class ReplicateRepository:
    def __init__(self):
        self.replicate = replicate

    def whisper(self, audio_url: str) -> str:
        output = self.replicate.run(
            "openai/whisper:4d50797290df275329f202e48c76360b3f22b08d28c196cbc54600319435f8d2",
            input={
                "audio": audio_url,
                "model": "large-v3",
            },
        )
        return output
