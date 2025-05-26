# Video Extraction Tool

This tool processes videos, extracts scenes, and uploads the processed content to Supabase storage.

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`)
- Supabase account with API credentials
- Google AI API key for Multimodal Embeddings

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GOOGLE_AI_API_KEY=your_google_ai_api_key
```

## New Scene Analysis Method

The tool now uses an advanced clustering-based scene analysis method with the following features:

### 🎯 Key Features

1. **Google Multimodal Embeddings**: Uses Google's state-of-the-art multimodal embeddings for superior frame representation
2. **FAISS Vector Database**: Efficient similarity search with video-partitioned storage
3. **K-means Clustering**: Automatic scene detection based on visual similarity
4. **Outlier Removal**: Intelligent filtering of anomalous frames
5. **Smart Scene Merging**: Combines short scenes for better coherence

### 🔧 Technical Implementation

- **Frame Sampling**: Extracts 1 frame per second from video
- **Embedding Generation**: 1408-dimensional Google Multimodal Embeddings
- **Vector Storage**: FAISS IndexFlatIP for cosine similarity search
- **Clustering**: Automatic optimal cluster number detection using silhouette score
- **Scene Generation**: Temporal grouping of similar visual content

### 📊 Performance Benefits

- **Accuracy**: Superior scene boundary detection compared to traditional methods
- **Scalability**: FAISS enables fast similarity search across large video collections
- **Flexibility**: Video-partitioned storage allows efficient per-video operations
- **Caching**: Intelligent caching system reduces reprocessing time

## Usage

### Simplified Command (Recommended)

Run the tool using the simple wrapper script:

```bash
./extract.py -v PATH_TO_YOUR_VIDEO
```

### Testing the New Scene Analyzer

Test the new Google Multimodal + FAISS scene analyzer:

```bash
# Set your Google AI API key
export GOOGLE_AI_API_KEY="your_api_key_here"

# Run the test script
python test_scene_analyzer.py
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

1. **Video Preprocessing**: Frame extraction at 1 FPS
2. **Embedding Generation**: Google Multimodal Embeddings for each frame
3. **Vector Storage**: FAISS database with video partitioning
4. **Clustering Analysis**: K-means clustering with outlier removal
5. **Scene Generation**: Temporal grouping and smart merging
6. **Content Upload**: Processed files to Supabase storage

## Output

All processed files, including:

- Preprocessed video (video.mp4)
- Extracted audio (audio.wav)
- Scene information (scenes.json) - **Now with cluster-based analysis**
- Video metadata (metadata.json)
- FAISS vector database (per-project partitioned)

are uploaded to a Supabase bucket named after the video ID.

# Attenz AI Project

## Setup and Testing

### 파이썬 워킹 디렉토리 설정하기

1. **터미널에서 프로젝트 루트로 이동하기**

   ```bash
   cd /path/to/attenz-ai
   ```

2. **환경 변수 설정하기**

   ```bash
   export GOOGLE_AI_API_KEY="your_google_ai_api_key"
   ```

3. **테스트 스크립트 실행하기**
   ```bash
   python test_scene_analyzer.py  # New Google Multimodal + FAISS test
   python test_pipeline.py        # Original pipeline test
   ```

### 새로운 Scene Analyzer 테스트

```bash
# Google AI API 키 설정
export GOOGLE_AI_API_KEY="your_api_key_here"

# 새로운 scene analyzer 테스트
python test_scene_analyzer.py

# 결과 확인
ls test_scenes_google_*.json
ls test_faiss_vector_db/
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
│       │   ├── scene_processor.py
│       │   ├── scene_analyzer.py          # 🆕 New Google Multimodal + FAISS
│       │   ├── google_embeddings.py       # 🆕 Google Multimodal Embeddings
│       │   └── vector_db.py               # 🆕 FAISS Vector Database
│       └── repository/
│           ├── __init__.py
│           ├── ai_repository.py
│           └── supabase_repository.py
├── test_scene_analyzer.py                 # 🆕 New scene analyzer test
└── test_pipeline.py
```

## Dependencies

### New Dependencies for Advanced Scene Analysis

- `faiss-cpu`: Efficient similarity search and clustering
- `google-generativeai`: Google Multimodal Embeddings API
- `google-cloud-aiplatform`: Alternative Vertex AI support
- `scikit-learn`: Machine learning utilities for clustering

### Installation

```bash
pip install -r requirements.txt
```

## API Keys and Configuration

### Google AI API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set the environment variable:
   ```bash
   export GOOGLE_AI_API_KEY="your_api_key_here"
   ```

### Supabase Configuration

1. Create a Supabase project
2. Get your project URL and API key
3. Set the environment variables:
   ```bash
   export SUPABASE_URL="your_supabase_url"
   export SUPABASE_KEY="your_supabase_key"
   ```
