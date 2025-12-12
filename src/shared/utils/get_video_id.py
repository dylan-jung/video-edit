import hashlib
import os


def get_video_id(video_path: str) -> str:
    """
    Create a unique video ID based on the video path.
    """
    sample_size = 4 * 1024  # 4KB
    try:
        with open(video_path, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end of file
            file_size = f.tell()
            f.seek(0)  # Reset to beginning
            
            # Read entire file if smaller than sample_size, otherwise read sample_size
            read_size = min(file_size, sample_size)
            file_sample = f.read(read_size)
    except (IOError, FileNotFoundError):
        # Fallback to path-based hash if file can't be read
        file_sample = os.path.abspath(video_path).encode('utf-8')
    
    return hashlib.sha256(file_sample).hexdigest()[:16]