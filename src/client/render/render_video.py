import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def time_to_seconds(time_str):
    """HH:MM:SS 형식의 시간을 초로 변환"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def seconds_to_time(seconds):
    """초를 HH:MM:SS 형식으로 변환"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def check_ffmpeg():
    """FFmpeg가 설치되어 있는지 확인"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_segment(input_video, trim_in, trim_out, output_path):
    """비디오에서 특정 구간을 추출"""
    trim_in_sec = time_to_seconds(trim_in)
    trim_out_sec = time_to_seconds(trim_out)
    duration = trim_out_sec - trim_in_sec
    
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-ss', str(trim_in_sec),
        '-t', str(duration),
        '-c', 'copy',  # 빠른 처리를 위해 재인코딩 없이 복사
        '-avoid_negative_ts', 'make_zero',
        '-y',  # 기존 파일 덮어쓰기
        output_path
    ]
    
    print(f"추출 중: {trim_in} ~ {trim_out} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"오류: {result.stderr}")
        return False
    
    return True


def create_concat_file(segments, concat_file_path):
    """FFmpeg concat 파일 생성"""
    with open(concat_file_path, 'w') as f:
        for segment in segments:
            f.write(f"file '{segment}'\n")

def get_video_path(project_id, video_id):
    info_path = f"./projects/{project_id}/{video_id}/info.json"
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return info[video_id]

def render_video(editing_state_path, output_path=None):
    """메인 렌더링 함수"""
    
    # FFmpeg 확인
    if not check_ffmpeg():
        print("오류: FFmpeg가 설치되어 있지 않습니다.")
        print("설치 방법: brew install ffmpeg (macOS) 또는 apt install ffmpeg (Ubuntu)")
        return False
    
    # JSON 파일 읽기
    try:
        with open(editing_state_path, 'r', encoding='utf-8') as f:
            editing_state = json.load(f)
    except FileNotFoundError:
        print(f"오류: {editing_state_path} 파일을 찾을 수 없습니다.")
        return False
    except json.JSONDecodeError:
        print(f"오류: {editing_state_path} 파일의 JSON 형식이 올바르지 않습니다.")
        return False
    
    project_id = editing_state.get('project_id', 'unknown')
    tracks = editing_state.get('tracks', [])

    if not tracks:
        print("오류: 트랙 정보가 없습니다.")
        return False
    
    # 출력 파일명 설정
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"rendered_{project_id}_{timestamp}.mp4"
    
    # 프로젝트 디렉토리 경로
    project_dir = Path(editing_state_path).parent
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        
        # 각 트랙을 개별 세그먼트로 추출
        for i, track in enumerate(tracks):
            src = track['src']
            trim_in = track['trimIn']
            trim_out = track['trimOut']
            
            # 소스 비디오 파일 경로
            src_video_path = get_video_path(project_id, src)
            
            # 세그먼트 파일 경로
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            
            # 세그먼트 추출
            if not extract_segment(str(src_video_path), trim_in, trim_out, segment_path):
                return False
            
            segment_files.append(segment_path)
        
        # concat 파일 생성
        concat_file_path = os.path.join(temp_dir, 'concat_list.txt')
        create_concat_file(segment_files, concat_file_path)
        
        # 최종 비디오 합치기
        print(f"최종 비디오 렌더링 중: {output_path}")
        concat_cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file_path,
            '-c', 'copy',
            '-y',
            output_path
        ]
        
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"오류: {result.stderr}")
            return False
    
    print(f"렌더링 완료: {output_path}")
    
    # 파일 정보 출력
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"파일 크기: {file_size:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='비디오 편집 상태를 기반으로 비디오를 렌더링합니다.')
    # parser.add_argument('editing_state', help='editing_state.json 파일 경로')
    # parser.add_argument('-o', '--output', help='출력 비디오 파일 경로')
    # parser.add_argument('-v', '--verbose', action='store_true', help='상세한 출력')
    
    args = parser.parse_args()
    
    args.editing_state = "projects/test/editing_state.json"
    args.output = "projects/test/rendered_video.mp4"

    print(f"편집 상태 파일: {args.editing_state}")
    print(f"출력 파일: {args.output}")
    
    success = render_video(args.editing_state, args.output)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main() 