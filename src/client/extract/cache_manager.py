import hashlib
import os
import tempfile

# Create a temporary directory for cache
CACHE_DIR = tempfile.gettempdir()

def get_cache_path(video_path: str, extension: str, args: dict) -> tuple[bool, str]:
    """
    Generate a cache path for a resized video based on sample of its content and target size.

    Args:
        video_path: Original video file path.
        extension: File extension for the cached file.
        args: Dictionary of arguments.

    Returns:
        A tuple (exists, filepath) where:
            - exists: Boolean indicating if the cache file already exists
            - filepath: Path under temp directory where the resized video will be stored
    """
    cache_dir = CACHE_DIR
    
    # Read a sample of the video file (first 64KB)
    sample_size = 64 * 1024  # 64KB
    try:
        with open(video_path, 'rb') as f:
            file_sample = f.read(sample_size)
    except (IOError, FileNotFoundError):
        # Fallback to path-based hash if file can't be read
        file_sample = os.path.abspath(video_path).encode('utf-8')
    
    # Create a unique key from file sample + arguments
    args_str = str(args).encode('utf-8')
    h = hashlib.sha256(file_sample + args_str).hexdigest()[:16]

    cache_path = os.path.join(cache_dir, f"{h}.{extension}")
    
    if os.path.exists(cache_path):
        return True, cache_path
    else:
        return False, cache_path
    
def clear_cache():
    """
    Clear all cached files in the temporary directory.
    """
    cache_dir = CACHE_DIR
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))    