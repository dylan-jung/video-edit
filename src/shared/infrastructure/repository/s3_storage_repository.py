from src.shared.infrastructure.repository.repository import Repository


class S3StorageRepository(Repository):
    def __init__(selt):
        self.s3_client = s3_client