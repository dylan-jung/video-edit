# Video Extraction Tool

This tool processes videos, extracts scenes, and uploads the processed content to Supabase storage.

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`)
- Supabase account with API credentials
- Google AI API key for Multimodal Embeddings
- OpenAI API key for GPT Vision analysis

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GOOGLE_AI_API_KEY=your_google_ai_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Scene Analysis Methods

The tool now supports multiple advanced scene analysis methods:

### 🎯 Google Multimodal + FAISS Method

1. **Google Multimodal Embeddings**: Uses Google's state-of-the-art multimodal embeddings for superior frame representation
2. **FAISS Vector Database**: Efficient similarity search with video-partitioned storage
3. **K-means Clustering**: Automatic scene detection based on visual similarity
4. **Outlier Removal**: Intelligent filtering of anomalous frames
5. **Smart Scene Merging**: Combines short scenes for better coherence

### 🧠 GPT Vision Analysis Method (NEW)

1. **OpenAI GPT-4 Vision**: Advanced visual understanding with natural language descriptions
2. **LangChain Integration**: Structured interaction with OpenAI APIs
3. **Frame Sampling**: Intelligent frame extraction and base64 encoding
4. **Scene Description**: Rich contextual analysis including objects, actions, emotions
5. **JSON Structured Output**: Detailed scene metadata with timestamps

#### 🔧 GPT Vision Technical Implementation

- **Frame Extraction**: OpenCV-based frame sampling (configurable rate)
- **Base64 Encoding**: Efficient image encoding for API transmission
- **Token Optimization**: Smart frame sampling to stay within token limits
- **LangChain ChatOpenAI**: Structured model interaction with proper error handling
- **JSON Parsing**: Robust response parsing with multiple format support

#### 📊 GPT Vision Benefits

- **Rich Descriptions**: Natural language scene understanding
- **Context Awareness**: Understands relationships between objects and actions
- **Emotion Detection**: Identifies emotional states and tone
- **OCR Capabilities**: Extracts visible text from video frames
- **Flexible Prompting**: Custom prompts for specific analysis needs

## Usage

### Simplified Command (Recommended)

Run the tool using the simple wrapper script:

```bash
./extract.py -v PATH_TO_YOUR_VIDEO
```

### Testing Scene Analyzers

#### Test Google Multimodal + FAISS Scene Analyzer

```bash
# Set your Google AI API key
export GOOGLE_AI_API_KEY="your_api_key_here"

# Run the test script
python test_scene_analyzer.py
```

#### Test GPT Vision Scene Analyzer (NEW)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Run the GPT scene analyzer test
python test_gpt_scene_analysis.py

# For batch testing multiple videos
python test_gpt_scene_analysis.py batch
```

### Using GPT Scene Analyzer in Code

```python
from src.server.indexing.gpt_scene_analyzer import analyze_video_with_gpt, analyze_video_with_custom_prompt

# Basic scene analysis with default prompt
result = analyze_video_with_gpt(
    video_path="your_video.mp4",
    chunk_index=0,
    model_name="gpt-4o"
)

# Custom prompt analysis
custom_result = analyze_video_with_custom_prompt(
    video_path="your_video.mp4",
    custom_prompt="Describe this video in detail",
    model_name="gpt-4o"
)
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
2. **Scene Analysis** (Choose one method):
   - **Google Multimodal**: Embedding generation + FAISS clustering
   - **GPT Vision**: AI-powered visual understanding + structured description
3. **Vector Storage**: FAISS database with video partitioning (Google method)
4. **Scene Generation**: Temporal grouping and smart merging
5. **Content Upload**: Processed files to Supabase storage

## Output

All processed files, including:

- Preprocessed video (video.mp4)
- Extracted audio (audio.wav)
- Scene information (scenes.json) - **Now with multiple analysis methods**
- Video metadata (metadata.json)
- FAISS vector database (per-project partitioned) - For Google method
- GPT analysis results (JSON format) - For GPT method

are uploaded to a Supabase bucket named after the video ID.

# Attenz AI Project

## 🚀 Quick Start - Web Interface

### Gradio Web Interface (NEW) 🌐

The easiest way to interact with the Attenz AI Agent is through the web interface:

```bash
# Install dependencies (includes gradio)
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_AI_API_KEY="your_google_ai_api_key"

# Launch the web interface
python run_gradio.py
```

Then open your browser and go to: **http://localhost:7860**

#### 🎯 Features

- **💬 Chat Interface**: Natural conversation with the AI agent
- **🔧 Tool Integration**: Automatic access to video analysis, search, and more
- **📱 Responsive Design**: Works on desktop and mobile
- **🎨 Modern UI**: Clean, intuitive interface with examples
- **🔄 Session Management**: Clear chat and reset conversations
- **⚡ Real-time**: Instant responses with progress indicators

#### 🛠️ Web Interface Capabilities

- **Video Analysis**: "Analyze the scene in video X"
- **Smart Search**: "Find videos with cars" or "Search for outdoor scenes"
- **Data Insights**: "Show me patterns in my video data"
- **General Assistant**: "What can you help me with?"

## Setup and Testing

### 파이썬 워킹 디렉토리 설정하기

1. **터미널에서 프로젝트 루트로 이동하기**

   ```bash
   cd /path/to/attenz-ai
   ```

2. **환경 변수 설정하기**

   ```bash
   export GOOGLE_AI_API_KEY="your_google_ai_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

3. **테스트 스크립트 실행하기**
   ```bash
   python test_scene_analyzer.py          # Google Multimodal + FAISS test
   python test_gpt_scene_analysis.py      # GPT Vision scene analysis test
   python test_pipeline.py                # Original pipeline test
   ```

### 새로운 Scene Analyzer 테스트

#### Google Multimodal + FAISS Method

```bash
# Google AI API 키 설정
export GOOGLE_AI_API_KEY="your_api_key_here"

# 새로운 scene analyzer 테스트
python test_scene_analyzer.py

# 결과 확인
ls test_scenes_google_*.json
ls test_faiss_vector_db/
```

#### GPT Vision Method (NEW)

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY="your_openai_api_key_here"

# GPT scene analyzer 테스트
python test_gpt_scene_analysis.py

# 결과 확인
ls gpt_analysis_result_*.json
ls gpt_custom_analysis_*.txt
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
│       │   ├── scene_analyzer.py          # Google Multimodal + FAISS
│       │   ├── gpt_scene_analyzer.py      # 🆕 GPT Vision + LangChain
│       │   ├── gemini_scene_analyzer.py   # Gemini Vision
│       │   ├── google_embeddings.py       # Google Multimodal Embeddings
│       │   └── vector_db.py               # FAISS Vector Database
│       └── repository/
│           ├── __init__.py
│           ├── ai_repository.py
│           └── supabase_repository.py
├── test_scene_analyzer.py                 # Google Multimodal + FAISS test
├── test_gpt_scene_analysis.py             # 🆕 GPT Vision scene analysis test
└── test_pipeline.py
```

## Dependencies

### Dependencies for Advanced Scene Analysis

- `faiss-cpu`: Efficient similarity search and clustering
- `google-generativeai`: Google Multimodal Embeddings API
- `langchain-openai`: LangChain OpenAI integration for GPT Vision
- `langchain-core`: Core LangChain components
- `opencv-python`: Video processing and frame extraction

### Required Python Packages

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages include:

- `opencv-python`: Video frame extraction
- `langchain-openai`: GPT Vision integration
- `faiss-cpu`: Vector similarity search
- `google-generativeai`: Google AI services
- `numpy`: Numerical computations
- `supabase`: Database operations
