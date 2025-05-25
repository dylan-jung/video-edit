import json
import os


def initialize_project(project_path: str):
    if os.path.exists(project_path):
        return

    with open(project_path, "w", encoding="utf-8") as f:
        json.dump({
            "timeline": {
                "clips": []
            }
        }, f, ensure_ascii=False)

