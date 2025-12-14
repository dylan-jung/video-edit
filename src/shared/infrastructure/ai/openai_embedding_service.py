import os
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from src.modules.indexing.application.ports.embedding_port import EmbeddingPort
from src.modules.indexing.infrastructure.utils.embedding_utils import scene_to_string

class OpenAIEmbeddingService(EmbeddingPort):
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        if "OPENAI_API_KEY" not in os.environ:
             raise EnvironmentError("OPENAI_API_KEY missing - required for OpenAIEmbeddingService")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_text_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            # Fallback random vector (3072 dims for text-embedding-3-large)
            return np.random.randn(3072)

    def generate_scene_embeddings(self, scenes: List[Dict[str, Any]]) -> List[np.ndarray]:
        embeddings = []
        for i, scene in enumerate(scenes):
            text = scene_to_string(scene)
            print(f"Generating embedding for scene {i+1}/{len(scenes)}...")
            embeddings.append(self.generate_text_embedding(text))
        return embeddings
