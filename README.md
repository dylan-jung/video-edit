# Video Extraction Tool

This tool processes videos, extracts scenes, and uploads the processed content to Supabase storage.

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`)
- Supabase account with API credentials

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Usage

### Simplified Command (Recommended)

Run the tool using the simple wrapper script:

```bash
./extract.py -v PATH_TO_YOUR_VIDEO
```

### Alternative Methods

Or use the original module path:

```bash
python -m src.client.extract.__index__ --video_path PATH_TO_YOUR_VIDEO
```

With the short form:

```bash
python -m src.client.extract.__index__ -v PATH_TO_YOUR_VIDEO
```

## Processing Steps

The tool performs the following operations:

1. Video and audio preprocessing (noise reduction, resizing)
2. Scene detection and division
3. Uploading processed files to Supabase storage

## Output

All processed files, including:

- Preprocessed video (video.mp4)
- Extracted audio (audio.wav)
- Scene information (scenes.json)
- Video metadata (metadata.json)

are uploaded to a Supabase bucket named after the video ID.

# Attenz AI Project

## Setup and Testing

### 파이썬 워킹 디렉토리 설정하기

1. **터미널에서 프로젝트 루트로 이동하기**

   ```bash
   cd /path/to/attenz-ai
   ```

2. **테스트 스크립트 실행하기**
   ```bash
   python test_pipeline.py
   ```

### 모듈 가져오기 오류 해결 방법

다음 방법 중 하나를 사용할 수 있습니다:

1. **sys.path를 통해 설정 (코드에 포함됨)**

   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.abspath(__file__)))
   ```

2. **PYTHONPATH 환경변수 설정**

   ```bash
   # 리눅스/맥
   export PYTHONPATH=/path/to/attenz-ai:$PYTHONPATH

   # 윈도우
   set PYTHONPATH=C:\path\to\attenz-ai;%PYTHONPATH%
   ```

3. **IDE 설정에서 Source Path 추가**
   - VS Code: settings.json에 python.analysis.extraPaths 설정
   - PyCharm: 프로젝트 구조에서 Content Root 설정

### 프로젝트 구조

```
attenz-ai/
├── src/
│   ├── __init__.py
│   └── server/
│       ├── __init__.py
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   └── scene_processor.py
│       └── repository/
│           ├── __init__.py
│           ├── ai_repository.py
│           └── supabase_repository.py
└── test_pipeline.py
```
