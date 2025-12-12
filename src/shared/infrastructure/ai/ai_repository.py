from .replicate_repository import ReplicateRepository


class AIRepository:
    def __init__(self):
        self.replicate = ReplicateRepository()

    def whisper(self, audio_input) -> str:
        return self.replicate.whisper(audio_input)
