import base64
import json  # Added import
import math
import multiprocessing
import os
import time  # Added for timing analysis
from collections import deque

import cv2
import matplotlib.pyplot as plt  # Added for plotting
import numpy as np


def _process_video_chunk(video_path, start_frame, end_frame):
    """Processes a chunk of video frames to calculate differences and their indices."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Worker] Error opening video: {video_path}")
        return []

    # Try to seek to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    actual_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if actual_start != start_frame:
        print(f"[Worker] Warning: Seeked to {actual_start} instead of {start_frame}")

    frame_diff_data = [] # Store tuples of (frame_index, diff)
    gray_prev = None
    current_frame_idx = actual_start

    # Read the frame *before* the loop starts to initialize gray_prev if possible
    # This ensures the first difference corresponds to actual_start + 1
    if current_frame_idx > 0 and start_frame == actual_start: # Avoid reading frame 0 if chunk starts there
         # If seeking was accurate and we are not at frame 0, read the previous frame
         cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start - 1)
         ret_prev, frame_prev = cap.read()
         if ret_prev:
             gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
         # Reset position back to start frame for the main loop
         cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_prev is not None:
            diff = cv2.absdiff(gray, gray_prev)
            mean_diff = np.mean(diff)
            # Store the diff associated with the *current* frame index
            frame_diff_data.append((current_frame_idx, mean_diff))

        gray_prev = gray
        current_frame_idx += 1

    cap.release()
    return frame_diff_data

def suggest_diff_threshold(video_path, percentile=85, num_workers=None, plot_diffs=False):
    """
    비디오 전체를 병렬로 분석하여 적절한 diff_threshold 값을 제안하고, 선택적으로 차이 그래프를 표시합니다.

    Parameters:
        video_path (str): 입력 비디오 파일 경로
        percentile (int): 프레임 차이 분포에서 임계값을 결정하는 데 사용할 백분위수
        num_workers (int, optional): 사용할 워커 프로세스 수. None이면 CPU 코어 수를 사용.
        plot_diffs (bool): True이면 프레임 차이 그래프를 표시합니다.
    Returns:
        suggested_threshold (float): 제안된 diff_threshold 값. 실패 시 None 반환.
    """
    print(f"\nAnalyzing video in parallel for threshold suggestion...")
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없어 임계값을 제안할 수 없습니다.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if total_frames < 2:
        print("비디오 프레임이 너무 적어 임계값을 제안할 수 없습니다.")
        return None

    # Handle potential invalid FPS
    valid_fps = fps is not None and fps > 0
    if not valid_fps:
        print("Warning: Could not determine valid FPS. Plot x-axis will be frame indices.")

    if num_workers is None:
        num_workers = os.cpu_count()
        print(f"Using default {num_workers} workers (CPU cores).")
    else:
        num_workers = max(1, num_workers)
        print(f"Using {num_workers} workers.")

    num_workers = min(num_workers, total_frames - 1)

    chunk_size = math.ceil(total_frames / num_workers)
    tasks = []
    for i in range(num_workers):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, total_frames)
        if end_frame > start_frame:
            tasks.append((video_path, start_frame, end_frame))

    if not tasks:
         print("No valid processing tasks created.")
         return None

    print(f"Divided {total_frames} frames into {len(tasks)} chunks for processing.")

    all_frame_diff_data = []
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(_process_video_chunk, tasks)

        for chunk_data in results:
            all_frame_diff_data.extend(chunk_data)

    except Exception as e:
        print(f"Error during parallel processing: {e}")
        return None

    analysis_duration = time.time() - start_time
    print(f"Parallel analysis finished in {analysis_duration:.2f}s. Found {len(all_frame_diff_data)} frame differences.")

    if not all_frame_diff_data:
        print("프레임 차이를 계산할 수 없었습니다 (아마도 비디오 문제?).")
        return None

    # Sort data by frame index for plotting and consistent processing
    all_frame_diff_data.sort(key=lambda x: x[0])

    # --- Plotting ---
    if plot_diffs:
        print("Generating frame difference plot...")
        try:
            frame_indices = [idx for idx, diff in all_frame_diff_data]
            diff_values = [diff for idx, diff in all_frame_diff_data]

            if valid_fps:
                times = [idx / fps for idx in frame_indices]
                x_label = "Time (seconds)"
            else:
                times = frame_indices # Use frame indices if FPS is invalid
                x_label = "Frame Index"

            plt.figure(figsize=(12, 6))
            plt.plot(times, diff_values, marker='.', linestyle='-', markersize=2)
            plt.xlabel(x_label)
            plt.ylabel("Mean Frame Difference")
            plt.title("Frame Difference Over Time")
            plt.grid(True)
            # Optionally add percentile line
            if len(diff_values) > 0:
                percentile_value = np.percentile(diff_values, percentile)
                plt.axhline(y=percentile_value, color='r', linestyle='--', label=f'{percentile}th Percentile ({percentile_value:.2f})')
                plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error generating plot: {e}")
    # --- End Plotting ---

    # Extract just the differences for percentile calculation
    all_frame_diffs_values = [diff for idx, diff in all_frame_diff_data]

    # Calculate the suggested threshold based on percentile
    try:
        suggested_threshold = np.percentile(all_frame_diffs_values, percentile)
        print(f"Suggested diff_threshold (based on {percentile}th percentile): {suggested_threshold:.2f}")
        suggested_threshold = max(suggested_threshold, 1.0)
        return suggested_threshold
    except Exception as e:
         print(f"Error calculating percentile: {e}")
         return None

def _resize_and_encode_frame(frame, target_size=512):
    """Helper function to resize frame (max dim = target_size) and encode to base64."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            print(f"Warning: Skipping frame due to zero height or width ({h}x{w}).")
            return None

        max_dim = max(h, w)
        if max_dim == 0: # Should be caught above, but for safety
             print(f"Warning: Skipping frame due to zero max dimension.")
             return None

        ratio = target_size / max_dim
        target_w = int(w * ratio)
        target_h = int(h * ratio)

        # Ensure target dimensions are positive
        if target_w <= 0 or target_h <= 0:
             print(f"Warning: Skipping frame due to non-positive target dimensions after resize ({target_w}x{target_h}). Original: {w}x{h}, Ratio: {ratio}")
             return None

        resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        retval, buffer = cv2.imencode('.jpg', resized_frame)
        if retval:
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string
        else:
            print(f"Warning: Failed to encode resized frame to base64.")
            return None
    except Exception as e:
        print(f"Error resizing/encoding frame: {e}")
        return None


def extract_keyframes(video_path, window_size=10, diff_threshold=30, target_size=512):
    """
    비디오 파일에서 키프레임을 추출하고, 즉시 크기 조정 및 Base64 인코딩을 수행합니다.

    Parameters:
        video_path (str): 입력 비디오 파일 경로
        window_size (int): 이동 평균 계산에 사용할 윈도우 크기
        diff_threshold (float): 키프레임 결정을 위한 프레임 차이 임계값
        target_size (int): 리사이즈 시 목표 최대 크기 (가로 또는 세로 중 긴 쪽)
    Returns:
        base64_keyframes (list of str): Base64 인코딩된 리사이즈된 키프레임 리스트
        keyframe_indices (list of int): 키프레임의 원본 비디오 프레임 인덱스 리스트
        keyframe_times (list of float): 키프레임의 원본 비디오 시간(초) 리스트
    """
    print(f"\nStep 1: Extracting, Resizing (max={target_size}px), and Encoding keyframes from: {video_path}")
    print(f"Using parameters: window_size={window_size}, diff_threshold={diff_threshold:.2f}, target_size={target_size}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return [], [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not determine video FPS. Timestamps will be based on frame indices.")
        fps = 1 # Assign a default fps to avoid division by zero, timestamps will be frame numbers

    base64_keyframes = []
    keyframe_indices = []
    keyframe_times = []

    # Store original frames temporarily for potential keyframe selection
    frames_queue = deque(maxlen=window_size)
    # Keep track of gray frames and differences for detection logic
    gray_frames_queue = deque(maxlen=window_size)
    frame_diffs_queue = deque(maxlen=window_size)

    frame_index = 0
    gray_prev = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_index = frame_index
        frame_index += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Store the original frame temporarily
        frames_queue.append(frame.copy()) # Use copy
        gray_frames_queue.append(gray)

        if gray_prev is not None:
            diff = cv2.absdiff(gray, gray_prev)
            mean_diff = np.mean(diff)
            frame_diffs_queue.append(mean_diff)

            if len(frame_diffs_queue) == window_size:
                smoothed_diff = np.mean(frame_diffs_queue)

                if smoothed_diff > diff_threshold:
                    # The keyframe corresponds to the *start* of the window where change was detected
                    keyframe_idx_in_window = current_frame_index - window_size + 1
                    # Ensure index is valid and not already added
                    if keyframe_idx_in_window >= 0 and (not keyframe_indices or keyframe_indices[-1] != keyframe_idx_in_window):
                        # Get the original frame from the beginning of the queue
                        keyframe_img = frames_queue[0]
                        # Resize and encode the identified keyframe
                        base64_string = _resize_and_encode_frame(keyframe_img, target_size)
                        if base64_string:
                            base64_keyframes.append(base64_string)
                            keyframe_indices.append(keyframe_idx_in_window)
                            keyframe_times.append(keyframe_idx_in_window / fps)
                        # else: Error message printed in helper function

        gray_prev = gray

    cap.release()
    print("Finished processing frames.")

    # Handle the first frame - always include it if not already captured
    if not keyframe_indices or keyframe_indices[0] != 0:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if ret:
             # Avoid adding duplicate index 0 if it was already found
            if 0 not in keyframe_indices:
                base64_string = _resize_and_encode_frame(first_frame, target_size)
                if base64_string:
                    # Insert at the beginning
                    base64_keyframes.insert(0, base64_string)
                    keyframe_indices.insert(0, 0)
                    keyframe_times.insert(0, 0.0)
                    # Re-sorting isn't strictly necessary if we always insert at 0,
                    # but uncomment if adding frames out of order becomes possible
                    # sorted_data = sorted(zip(keyframe_indices, keyframe_times, base64_keyframes), key=lambda x: x[0])
                    # keyframe_indices = [idx for idx, _, _ in sorted_data]
                    # keyframe_times = [time for _, time, _ in sorted_data]
                    # base64_keyframes = [b64 for _, _, b64 in sorted_data]
        cap.release()

    return base64_keyframes, keyframe_indices, keyframe_times

def save_keyframes(base64_keyframes, keyframe_times, output_dir):
    """
    Base64 인코딩된 키프레임과 해당 시간을 각 프레임별 JSON 파일로 저장합니다.

    Parameters:
        base64_keyframes (list of str): Base64 인코딩된 키프레임 문자열 리스트
        keyframe_times (list of float): 키프레임 시간(초) 리스트
        output_dir (str): 저장할 디렉토리 경로
    """
    print(f"\nStep 2: Saving keyframes data as individual JSON files in {output_dir}...") # Step adjusted
    os.makedirs(output_dir, exist_ok=True)

    if not base64_keyframes:
        print("No keyframes data to save.")
        # Optionally create an empty indicator file or log
        with open(os.path.join(output_dir, "_no_keyframes.txt"), 'w') as f:
            f.write("No keyframes were extracted or transformed.\n")
        return

    if len(base64_keyframes) != len(keyframe_times):
        print(f"Warning: Mismatch between saved keyframes ({len(base64_keyframes)}) and times ({len(keyframe_times)}). This indicates an error during processing.")
        # Decide how to handle - maybe save only matching pairs?
        # For now, we proceed but limit saving to the shorter list length.
        min_len = min(len(base64_keyframes), len(keyframe_times))
        base64_keyframes = base64_keyframes[:min_len]
        keyframe_times = keyframe_times[:min_len]
        print(f"Proceeding to save the first {min_len} matching keyframes.")

    num_saved = 0
    for i, (time_sec, b64_str) in enumerate(zip(keyframe_times, base64_keyframes)):
        # Create data structure for JSON
        keyframe_data = {
            "index": i, # Use enumeration index for output file naming consistency
            # Consider storing original frame index if needed: "original_frame_index": keyframe_indices[i]
            "time_seconds": round(time_sec, 3), # Round time for cleaner output
            "base64_jpeg": b64_str
        }

        # Define filename using index
        filename = os.path.join(output_dir, f"keyframe_{i:04d}.json")

        # Save data to JSON file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(keyframe_data, f, indent=4) # Use indent for readability
            num_saved += 1
        except Exception as e:
            print(f"Error saving keyframe {i} to {filename}: {e}")
            # Continue to next frame if one fails

    print(f"Successfully saved data for {num_saved}/{len(base64_keyframes)} keyframes to {output_dir}")


# 예시 사용
if __name__ == "__main__":
    # video_file = "system_video.mp4"  # 비디오 파일 경로
    video_file = "test.MOV"  # 비디오 파일 경로
    output_dir = "keyframes_output_json" # Changed output dir name
    target_size = 512 # Renamed from target_width, used for max dimension
    window_size = 10  # window_size는 고정값 또는 사용자가 설정
    diff_threshold = 3 # 이전 값
    suggest_threshold_flag = True # Changed variable name for clarity
    plot_threshold_debug = True # Added flag to enable plotting

    # Step 0: Suggest threshold (Optional)
    if suggest_threshold_flag:
        # Decide on number of workers for suggestion, e.g., os.cpu_count()
        num_suggestion_workers = os.cpu_count() # Or set to a fixed number like 4
        suggested_threshold_val = suggest_diff_threshold(
            video_file,
            percentile=95,
            num_workers=num_suggestion_workers,
            plot_diffs=plot_threshold_debug # Pass the plotting flag
        )
        if suggested_threshold_val is not None: # Check if suggestion was successful
             diff_threshold = suggested_threshold_val # Use the suggested threshold
        else:
             print(f"Threshold suggestion failed, using default: {diff_threshold}")
    print(f"Using threshold: {diff_threshold:.2f}") # Updated print message

    print(f"\nStarting keyframe extraction and processing...")
    # Step 1: Extract, Resize, Encode Keyframes
    base64_keyframes, keyframe_indices, keyframe_times = extract_keyframes(
        video_file,
        window_size=window_size,
        diff_threshold=diff_threshold,
        target_size=target_size # Pass target_size here
    )

    if base64_keyframes:
        print(f"\nExtraction Summary: Found and processed {len(base64_keyframes)} keyframes.")
        if keyframe_indices:
            print(f"First keyframe original index: {keyframe_indices[0]}, time: {keyframe_times[0]:.3f}s")
            print(f"Last keyframe original index: {keyframe_indices[-1]}, time: {keyframe_times[-1]:.3f}s")

        # Step 2: Saving results (was Step 3)
        # Pass the directly obtained base64_keyframes and corresponding times
        save_keyframes(base64_keyframes, keyframe_times, output_dir)
    else:
        print("No keyframes extracted or processed with the used parameters.") # Updated message
