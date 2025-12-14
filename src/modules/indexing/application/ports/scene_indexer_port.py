from abc import ABC, abstractmethod

class SceneIndexerPort(ABC):
    @abstractmethod
    def run(self, project_id: str, video_id: str, scene_descriptions_path: str, vector_db_url: str) -> None:
        """
        Index scenes from descriptions.
        """
        pass
