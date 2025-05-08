import argparse
import base64
import glob
import json
import os

import cv2
import numpy as np


def decode_base64_image(base64_string):
    """Decodes a Base64 string into an OpenCV image."""
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        return None

def view_keyframes(json_dir):
    """Loads and displays keyframes from JSON files in a directory."""
    json_pattern = os.path.join(json_dir, "keyframe_*.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        print(f"No keyframe JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} keyframe files.")
    print("Controls: 'n' -> next, 'p' -> previous, 'q' -> quit")

    current_index = 0
    while True:
        json_path = json_files[current_index]
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            base64_str = data.get("base64_jpeg")
            time_sec = data.get("time_seconds", "N/A")
            frame_idx = data.get("index", current_index) # Use file index if not in JSON

            if not base64_str:
                print(f"Warning: 'base64_jpeg' not found or empty in {json_path}")
                img = np.zeros((100, 200, 3), dtype=np.uint8) # Placeholder for missing image
                cv2.putText(img, "No Image Data", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                img = decode_base64_image(base64_str)
                if img is None:
                    print(f"Warning: Failed to decode image from {json_path}")
                    img = np.zeros((100, 200, 3), dtype=np.uint8) # Placeholder
                    cv2.putText(img, "Decoding Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display the image
            window_title = f"Keyframe {frame_idx} (Time: {time_sec}s) - {os.path.basename(json_path)}"
            cv2.imshow(window_title, img)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF # Use waitKey(0) to wait indefinitely for a key

            # Close previous window if title changes (OpenCV doesn't easily rename)
            cv2.destroyWindow(window_title)

            if key == ord('q'):
                break
            elif key == ord('n'):
                current_index = (current_index + 1) % len(json_files)
            elif key == ord('p'):
                current_index = (current_index - 1 + len(json_files)) % len(json_files)
            # Allow closing with the window 'X' button
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                 break


        except FileNotFoundError:
            print(f"Error: File not found {json_path}")
            # Decide how to handle: skip, retry, or exit
            current_index = (current_index + 1) % len(json_files) # Simple skip
            if current_index == 0 and len(json_files) > 1: # Avoid infinite loop if first file missing
                print("Stopping due to error reading first file.")
                break
            elif len(json_files) <= 1:
                 break # Exit if only one file and it's missing
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            current_index = (current_index + 1) % len(json_files) # Simple skip
            continue # Skip corrupted JSON
        except Exception as e:
            print(f"An unexpected error occurred processing {json_path}: {e}")
            current_index = (current_index + 1) % len(json_files) # Simple skip
            continue

    cv2.destroyAllWindows()
    print("Exited keyframe viewer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View extracted keyframes stored as JSON files.")
    parser.add_argument("json_dir", help="Directory containing the keyframe JSON files (e.g., keyframes_output_json).")
    args = parser.parse_args()

    if not os.path.isdir(args.json_dir):
        print(f"Error: Directory not found: {args.json_dir}")
    else:
        view_keyframes(args.json_dir)

# Example Usage:
# python src/view_keyframe.py keyframes_output_json
