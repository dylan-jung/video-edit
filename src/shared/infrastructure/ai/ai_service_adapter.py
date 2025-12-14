from src.config.settings import get_settings
from src.shared.application.ports.ai_service import AIServicePort
import replicate


class AIServiceAdapter(AIServicePort):
    def __init__(self):
        self.replicate = replicate.Client(api_token=get_settings().REPLICATE_API_TOKEN)

    def _whisper(self, audio_input) -> str:
        output = self.replicate.run(
            "openai/whisper:4d50797290df275329f202e48c76360b3f22b08d28c196cbc54600319435f8d2",
            input={
                "audio": audio_input,
                "model": "large-v3",
            },
        )
        return output

    def transcribe_audio(self, audio_input) -> str:
        return self._whisper(audio_input)
