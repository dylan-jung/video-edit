import json
import os
import subprocess

import cv2

from .cache_manager import get_cache_path


def resize_and_cache_video(
    video_path,
    target_size=480,
    text_area_height=16,
    use_cache: bool = True
):
    """
    Resize video and cache it if target_size is provided.

    Args:
        video_path (str): Path to the input video.
        target_size (int): Pre-resize max dimension.
        text_area_height (int): Height of the black text area at the top in pixels.

    Returns:
        str: Path to the working video (original or resized cached version).
    """
    working_path = video_path
    if target_size:
        hit, cache_path = get_cache_path(video_path, 'mp4', {'target_size': target_size, 'text_area_height': text_area_height})
        if not hit or not use_cache:
            cap_m = cv2.VideoCapture(video_path)
            if not cap_m.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            ow = int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH))
            oh = int(cap_m.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_m.release()
            if ow >= oh:
                nw, nh = target_size, int(oh/ow*target_size)
            else:
                nh, nw = target_size, int(ow/oh*target_size)
            
            # Add black area at the top and place timestamp in it
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'scale={target_size}:-1:force_original_aspect_ratio=decrease,'
                       f'pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2:black,'
                       f'pad=iw:ih+{text_area_height}:0:{text_area_height}:black,'
                       f'drawtext=text=\'%{{pts\\:hms}}\':fontcolor=white:fontsize=12:boxborderw=5:x=(w-text_w)-3:y=3',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast',
                '-an', cache_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        working_path = cache_path
    return working_path

def extract_and_cache_audio(
    video_path: str,
    noise_reduction: bool = True,
    use_cache: bool = True
) -> str:
    """
    Extract audio and optionally apply FFmpeg's afftdn noise reduction.
    Results are cached to avoid reprocessing.

    Args:
        video_path (str): Path to the input video.
        output_audio_path (str): Path where to save the extracted audio.
        noise_reduction (bool): Whether to apply noise reduction.

    Returns:
        str: Path to saved audio file (wav or mp3).
    """
    hit, cache_path = get_cache_path(video_path, "wav", {'noise_reduction': noise_reduction})
    if not hit or not use_cache:
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vn']
        if noise_reduction:
            cmd += ['-af', 'afftdn']
        cmd += ['-ac', '1', '-ar', '16000']
        cmd += [cache_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    
    return cache_path

def extract_and_cache_video_metadata(video_path: str, use_cache: bool = True) -> str:
    """
    Extract video metadata including creation time using ffprobe.
    Results are cached to avoid reprocessing.
    
    Args:
        video_path (str): Path to the input video.
        
    Returns:
        dict: Dictionary containing video metadata with creation_time as one of the keys.
    """
    # Check if metadata is already cached
    hit, cache_path = get_cache_path(video_path, 'json', {'type': 'metadata'})
    if hit and use_cache:
        return cache_path
    
    # Get basic file info
    metadata = {
        'filename': os.path.basename(video_path),
        'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        'path': video_path,
        'creation_time': None
    }
    
    # Get creation_time
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-show_entries', 'format_tags=creation_time',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        creation_time = result.stdout.strip()
        metadata['creation_time'] = creation_time if creation_time else None
    except Exception:
        metadata['creation_time'] = None
    
    # Get video format information
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        # Extract format info
        if 'format' in info:
            format_info = info['format']
            metadata['format'] = format_info.get('format_name')
            metadata['duration'] = float(format_info.get('duration', 0))
            metadata['bit_rate'] = int(format_info.get('bit_rate', 0))
            
            # Get additional tags if available
            if 'tags' in format_info:
                tags = format_info['tags']
                for key, value in tags.items():
                    metadata[key.lower()] = value
        
        # Extract video stream info
        if 'streams' in info:
            for stream in info['streams']:
                if stream.get('codec_type') == 'video':
                    metadata['width'] = stream.get('width')
                    metadata['height'] = stream.get('height')
                    metadata['codec'] = stream.get('codec_name')
                    
                    # Calculate fps
                    if 'r_frame_rate' in stream:
                        try:
                            num, den = map(int, stream['r_frame_rate'].split('/'))
                            metadata['fps'] = num / den if den != 0 else 0
                        except:
                            metadata['fps'] = 0
                    break
    except Exception as e:
        metadata['error'] = str(e)
    
    # Cache the metadata
    with open(cache_path, 'w') as f:
        json.dump(metadata, f)
    
    return cache_path

def process_video_and_audio(
    input_path: str,
    target_size: int = 640,
    text_area_height: int = 16,
    noise_reduction: bool = True,
    extract_metadata: bool = False,
    use_cache: bool = True
) -> tuple:
    """
    Process a video by resizing it and extracting denoised audio.
    Automatically generates cached output paths.
    
    Args:
        input_path (str): Path to the input video.
        target_size (int): Video resize max dimension.
        text_area_height (int): Height of the text area in video.
        noise_reduction (bool): Whether to apply audio noise reduction.
        extract_metadata (bool): Whether to extract and return video metadata.
        
    Returns:
        tuple: (processed_video_path, processed_audio_path, metadata) if extract_metadata=True
               (processed_video_path, processed_audio_path) otherwise
    """
    
    # Process video
    processed_video = resize_and_cache_video(
        input_path, 
        target_size=target_size,
        text_area_height=text_area_height,
        use_cache=use_cache
    )
    
    # Process audio
    processed_audio = extract_and_cache_audio(
        input_path,
        noise_reduction=noise_reduction,
        use_cache=use_cache
    )
    
    # Extract metadata if requested
    if extract_metadata:
        metadata_path = extract_and_cache_video_metadata(input_path, use_cache=use_cache)
        return processed_video, processed_audio, metadata_path
    
    return processed_video, processed_audio
