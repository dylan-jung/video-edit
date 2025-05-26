import argparse

from src.server.indexing.pipeline import pipeline


def test_pipeline():
    args = argparse.ArgumentParser()
    args.add_argument("--project_id", "-p", type=str, required=True)
    args.add_argument("--video_id", "-v", type=str, required=True)
    args = args.parse_args()

    pipeline(args.project_id, args.video_id)

if __name__ == "__main__":
    test_pipeline()
