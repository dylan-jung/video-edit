import argparse
import os
import sys

from src.client.extract.pipeline import pipeline


def main():
    """
    Command line interface for video extraction.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Extract and process video content.')
    parser.add_argument('--video_path', '-v', type=str, required=True,
                        help='Path to the video file to be processed')
    parser.add_argument('--project_id', '-p', type=str, required=True,
                        help='Project ID')
    # Parse arguments
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found at {args.video_path}")
        sys.exit(1)

    print(f"🔍 Starting video extraction for: {os.path.basename(args.video_path)}")
    video_id = pipeline(args.project_id, args.video_path)
    print("✅ Video extraction completed successfully!")
    print("video_id: ", video_id)

if __name__ == "__main__":
    main()
