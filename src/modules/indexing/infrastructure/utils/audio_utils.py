import os
import subprocess
import tempfile
from datetime import timedelta
from typing import List, Tuple

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file using ffmpeg.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            'ffprobe', '-i', audio_path, 
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0

def split_audio_chunks(audio_path: str, chunk_duration: int = 10) -> List[Tuple[str, float, float]]:
    """
    Split audio file into chunks using ffmpeg.
    
    Args:
        audio_path: Path to the MP3 file
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of tuples (chunk_file_path, start_time, end_time)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get total duration
    total_duration = get_audio_duration(audio_path)
    print(f"Total audio duration: {total_duration:.2f} seconds")
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    print(f"Creating audio chunks in: {temp_dir}")
    
    chunks = []
    chunk_index = 0
    
    for start_time in range(0, int(total_duration), chunk_duration):
        end_time = min(start_time + chunk_duration, total_duration)
        
        # Generate chunk filename
        chunk_filename = f"chunk_{chunk_index:04d}.wav"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        # ffmpeg command to extract chunk
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-acodec', 'copy',
            '-y',  # Overwrite output files
            chunk_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            chunks.append((chunk_path, float(start_time), float(end_time)))
            chunk_index += 1
            # print(f"Created chunk: {chunk_filename} ({start_time}s - {end_time}s)") # Too verbose
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk {chunk_index}: {e}")
            continue
    
    print(f"Successfully created {len(chunks)} audio chunks")
    return chunks

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to hh:mm:ss.fff format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds_remainder = td.total_seconds() % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"
