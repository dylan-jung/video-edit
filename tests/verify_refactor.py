import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, ANY

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock IndexingService BEFORE importing WorkerService to avoid loading deep dependencies
sys.modules['src.app.services.indexing_service'] = MagicMock()
# Mock google.cloud to avoid init credentials error
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()

from src.repository.cloud_storage_repository import CloudStorageRepository
# We need to reload WorkerService or import it after mocking if it was already imported (it wasn't)
from src.app.services.worker_service import WorkerService
from src.app.services.queue.base import QueueService

def test_presigned_url_generation():
    print("Testing Pre-signed URL Generation...")
    repo = CloudStorageRepository()
    # Mocking bucket and blob because we don't have creds
    repo.bucket = MagicMock()
    blob_mock = MagicMock()
    repo.bucket.blob.return_value = blob_mock
    blob_mock.generate_signed_url.return_value = "https://example.com/upload-url"

    url = repo.generate_upload_signed_url("p1", "v1", "test.mp4", "video/mp4")
    print(f"Generated URL: {url}")
    assert url == "https://example.com/upload-url"
    
    # Check if method was called correctly (PUT)
    blob_mock.generate_signed_url.assert_called_with(
        expiration=ANY, 
        method="PUT", 
        content_type="video/mp4"
    )
    print("✅ Pre-signed URL Generation Test Passed")

class MockQueueService(QueueService):
    def send_message(self, message):
        pass
    def receive_messages(self, max_messages=10, wait_time_seconds=20):
        return []
    def delete_message(self, receipt_handle):
        pass

async def test_worker_logic():
    print("Testing Worker Logic...")
    
    mock_queue = MockQueueService()
    worker = WorkerService(queue_service=mock_queue)
    
    # Mock IndexingService.run_indexing
    with patch('src.app.services.worker_service.IndexingService.run_indexing', new_callable=AsyncMock) as mock_indexing:
        message = {
            'Body': '{"project_id": "p1", "video_id": "v1"}',
            'ReceiptHandle': 'handle123'
        }
        
        await worker.process_message(message)
        
        mock_indexing.assert_called_once_with("p1", "v1")
        print("✅ Worker processed message and called IndexingService")

# AsyncMock helper for python < 3.8 if needed, but simple enough here
from unittest.mock import AsyncMock

if __name__ == "__main__":
    test_presigned_url_generation()
    asyncio.run(test_worker_logic())
