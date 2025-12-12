from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingPort(ABC):
    @abstractmethod
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        """
        pass

    @abstractmethod
    def generate_scene_embeddings(self, scenes: List[dict]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of scene dictionaries.
        """
        pass
