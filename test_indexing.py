import json
import os
import sys

from src.server.indexing.pipeline import pipeline


def test_indexing():
    analysis_path = sys.argv[1]
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    embedding_generator = GoogleEmbeddingGenerator()
    embeddings = embedding_generator.generate_scene_embeddings(analysis)


