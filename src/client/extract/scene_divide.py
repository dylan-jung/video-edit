import json
import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .cache_manager import get_cache_path


def visualize_correlations(corrs, fps, threshold=None, noise_level=None, output_path=None, show=False):
    """
    Visualize frame-to-frame correlations with optional threshold line.

    Args:
        corrs (list): List of correlation values between consecutive frames.
        fps (float): Frames per second of the video.
        threshold (float, optional): Threshold value to highlight.
        noise_level (float, optional): Noise level estimate to highlight.
        output_path (str, optional): Path to save the visualization image.
        show (bool): Whether to display the plot immediately.

    Returns:
        str or None: Path to the saved image if output_path is provided, otherwise None.
    """
    # Create time axis in seconds
    time_axis = [i / fps for i in range(len(corrs))]

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, corrs, '-', linewidth=1, label='Frame Correlation')

    if noise_level is not None:
        plt.axhline(y=noise_level, color='g', linestyle=':',
                    label=f'Noise Level ({noise_level:.3f})')

    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--',
                    label=f'Threshold ({threshold:.3f})')

    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Correlation')
    plt.title('Frame-to-Frame Histogram Correlation')
    plt.grid(True)
    plt.legend()

    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(
            os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        return output_path

    if show:
        plt.show()

    plt.close()
    return None


def calculate_otsu_threshold(data, bins=256):
    """
    Calculate Otsu's threshold to separate two classes in data using NumPy.

    Args:
        data (np.array): 1D array of values to threshold
        bins (int): Number of histogram bins

    Returns:
        float: Otsu's threshold value
    """
    # Calculate histogram
    hist, edges = np.histogram(data, bins=bins)

    # Calculate bin midpoints
    mids = (edges[:-1] + edges[1:]) / 2

    # Calculate weights and means
    w0 = np.cumsum(hist)                       # Cumulative sum for weight 0
    w1 = np.cumsum(hist[::-1])[::-1]           # Cumulative sum for weight 1

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        m0 = np.cumsum(hist * mids) / w0       # Mean for class 0
        m1 = (np.cumsum((hist * mids)[::-1]) /
              w1[::-1])[::-1]  # Mean for class 1

    # Calculate between-class variance
    sigma_b2 = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:])**2

    # Find the index of maximum variance (handle potential NaN values)
    idx = np.nanargmax(sigma_b2)

    # Return threshold value
    return float(mids[idx])


def resize_and_cache_video(
    video_path,
    target_size=640,
    text_area_height=24
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
        cache_path = get_cache_path(video_path, 'mp4', {
                                    'target_size': target_size, 'text_area_height': text_area_height})
        if not os.path.exists(cache_path):
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
                f'drawtext=text=\'%{{pts\\:hms\\:%.3f}}\':fontcolor=white:fontsize=12:boxborderw=5:x=(w-text_w)-20:y=10',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast',
                '-an', cache_path
            ]
            # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            subprocess.run(cmd, check=True)
        working_path = cache_path
    return working_path


def detect_scene_boundaries(
    video_path,
    threshold=0.5,
    hist_size=(8, 8, 8),
    use_gpu=False,
    min_scene_length=2.0,
    auto_threshold=False,
    max_threshold=0.95,
    percentile=5.0,
    visualize=False,
    vis_output_path=None,
    bg_eval_seconds=2.0
):
    """
    Detect scene boundaries in a video using histogram comparison.

    Args:
        video_path (str): Path to the input video.
        threshold (float): Manual correlation threshold for scene change.
        hist_size (tuple): Histogram bins for H,S,V.
        use_gpu (bool): Use OpenCV CUDA.
        min_scene_length (float): Merge scenes shorter than this.
        auto_threshold (bool): If True, use improved auto-thresholding.
        percentile (float): Percentile for old auto thresholding method.
        visualize (bool): If True, visualize the correlations.
        vis_output_path (str, optional): Path to save visualization image.
        bg_eval_seconds (float): Seconds of video to use for background noise evaluation.

    Returns:
        List of (start_sec, end_sec) scenes with millisecond precision.
    """
    # Read frames and compute histograms
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    hists = []
    gpu_avail = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if gpu_avail:
            g = cv2.cuda_GpuMat()
            g.upload(frame)
            hsv = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2HSV)
            planes = cv2.cuda.split(hsv)
            gh = cv2.cuda.calcHist(planes, hist_size, [0, 180, 0, 256, 0, 256])
            cv2.cuda.normalize(gh, gh, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX)
            hists.append(gh.download().flatten())
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, hist_size, [
                                0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hists.append(hist.flatten())
    cap.release()

    # Correlation between consecutive frames
    corrs = [cv2.compareHist(hists[i], hists[i+1], cv2.HISTCMP_CORREL)
             for i in range(len(hists)-1)]

    # Determine threshold
    noise_level = None
    thr = threshold
    if auto_threshold:
        # 1. Calculate background noise from first bg_eval_seconds
        # Use at most 1/4 of video
        bg_frames = min(int(bg_eval_seconds * fps), len(corrs) // 4)
        if bg_frames > 0:
            bg_corrs = corrs[:bg_frames]
            mean_noise = np.mean(bg_corrs)
            std_noise = np.std(bg_corrs)

            # 2. Estimate noise floor as mean - std
            noise_level = mean_noise - std_noise

            # 3. Use Otsu's method to calculate threshold
            thr = calculate_otsu_threshold(corrs)

            # Ensure the threshold is below the noise level (more conservative)
            thr = min(thr, noise_level)
        else:
            # Fallback to percentile method if not enough frames
            thr = np.percentile(corrs, percentile)

    if thr > max_threshold:
        thr = max_threshold

    # Visualize correlations if requested
    if visualize:
        visualize_correlations(corrs, fps, thr, noise_level,
                               vis_output_path, show=(vis_output_path is None))

    # Detect boundaries
    boundaries = [0]
    for i, c in enumerate(corrs, 1):
        if c < thr:
            boundaries.append(i)
    boundaries.append(len(hists))
    # Convert to time and merge short scenes
    raw = [(boundaries[i]/fps, boundaries[i+1]/fps)
           for i in range(len(boundaries)-1)]
    merged = []
    for s, e in raw:
        if merged and (e-s) < min_scene_length:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Format timestamps with millisecond precision
    formatted_scenes = [(round(s, 3), round(e, 3)) for s, e in merged]
    return formatted_scenes


def segment_scenes_histogram(
    video_path,
    threshold=0.5,
    hist_size=(8, 8, 8),
    use_gpu=False,
    min_scene_length=2.0,
    target_size=None,
    auto_threshold=False,
    percentile=5.0,
    visualize=False,
    vis_output_path=None,
    bg_eval_seconds=2.0
):
    """
    Segment a video into scenes via histogram comparison, merge too-short scenes,
    with optional resizing, caching, and automatic thresholding by percentile.

    Args:
        video_path (str): Path to the input video.
        threshold (float): Manual correlation threshold for scene change.
        hist_size (tuple): Histogram bins for H,S,V.
        use_gpu (bool): Use OpenCV CUDA.
        min_scene_length (float): Merge scenes shorter than this.
        target_size (int): Pre-resize max dimension.
        auto_threshold (bool): If True, use improved auto-thresholding.
        percentile (float): Percentile for old auto thresholding method.
        visualize (bool): If True, visualize the correlations.
        vis_output_path (str, optional): Path to save visualization image.
        bg_eval_seconds (float): Seconds of video to use for background noise evaluation.

    Returns:
        List of (start_sec, end_sec) scenes.
    """
    # Prepare video (resized & cached)
    working_path = resize_and_cache_video(video_path, target_size)

    # Analyze video and detect scenes
    return detect_scene_boundaries(
        working_path,
        threshold,
        hist_size,
        use_gpu,
        min_scene_length,
        auto_threshold,
        percentile,
        visualize,
        vis_output_path,
        bg_eval_seconds
    )


def split_video(
    video_path,
    scenes,
    output_dir,
    ffmpeg_path='ffmpeg',
    use_cache=False
):
    """
    Split video into segments and save via ffmpeg, with optional caching.

    Args:
        video_path (str): Input video path.
        scenes (list): List of (start_sec, end_sec).
        output_dir (str): Directory to save segments.
        ffmpeg_path (str): FFmpeg executable.
        use_cache (bool): If True, skip already existing output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    for i, (s, e) in enumerate(scenes, 1):
        out = os.path.join(output_dir, f"{base}_scene{i:03d}.mp4")
        if use_cache and os.path.exists(out):
            print(f"Skipping scene {i}, cached: {out}")
            continue
        cmd = [ffmpeg_path, '-y', '-ss',
               str(s), '-to', str(e), '-i', video_path, '-c', 'copy', out]
        print('Running:', ' '.join(cmd))
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode:
            print(f"Error scene {i}: {r.stderr.decode()}")
        else:
            print(f"Saved scene {i}: {out}")


def save_scene_info(video_path, scenes, cache=True):
    """
    Save scene information to a JSON file.

    Args:
        video_path (str): Input video path.
        scenes (list): List of (start_sec, end_sec) scenes.
        cache (bool): If True, cache the scene information.

    Returns:
        str: Path to the saved scene information file.
    """

    base = os.path.splitext(os.path.basename(video_path))[0]

    scene_info = []
    for i, (s, e) in enumerate(scenes, 1):
        scene_info.append({
            "scene_number": i,
            "start_time": s,
            "end_time": e,
            "duration": round((e-s), 3),
            # "filename": f"{base}.mp4"
        })

    if cache:
        hit, scene_json_path = get_cache_path(
            video_path, 'json', {'scenes': scene_info})
        if not hit:
            with open(scene_json_path, 'w') as f:
                f.write(json.dumps(scene_info, indent=4))
        print(f"Saved scene information to: {scene_json_path}")
        return scene_json_path
    else:
        return scene_info
