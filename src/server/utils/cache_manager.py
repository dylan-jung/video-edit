import hashlib
import os
import tempfile

# Create a temporary directory for cache
CACHE_DIR = tempfile.gettempdir()

def get_file_id(file_path: str) -> str:
    """
    Create a unique file ID based on the file path and a sample of its content.
    Works with any file type, not just videos.
    """
    sample_size = 64 * 1024  # 64KB
    try:
        with open(file_path, 'rb') as f:
            # Read up to sample_size bytes, may be less if file is smaller
            file_sample = f.read(sample_size)
    except (IOError, FileNotFoundError):
        # Fallback to path-based hash if file can't be read
        file_sample = os.path.abspath(file_path).encode('utf-8')
    
    return hashlib.sha256(file_sample).hexdigest()[:16]

def get_cache_path(file_path: str, args: dict) -> tuple[bool, str]:
    """
    Generate a cache path for a processed file based on sample of its content and processing arguments.
    Works with any file type, not just videos.

    Args:
        file_path: Original file path.
        args: Dictionary of arguments used for processing.

    Returns:
        A tuple (exists, filepath) where:
            - exists: Boolean indicating if the cache file already exists
            - filepath: Path under temp directory where the processed file will be stored
    """
    cache_dir = CACHE_DIR
    
    # Get file ID using the existing function
    file_id = get_file_id(file_path)
    
    # Create a unique key from file ID + arguments
    args_str = str(args).encode('utf-8')
    h = hashlib.sha256((file_id.encode('utf-8') + args_str)).hexdigest()[:16]

    # Use the original file extension
    _, original_extension = os.path.splitext(file_path)
    if not original_extension:
        original_extension = ".dat"  # Default extension if none exists
    
    cache_path = os.path.join(cache_dir, f"{h}{original_extension}")
    
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