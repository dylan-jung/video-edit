#!/bin/bash

# Video Rendering Script for Projects
# 사용법: ./render_project.sh [project_name] [output_file]

# 기본 설정
PROJECTS_DIR="projects"
SCRIPT_NAME="render_video.py"

# 도움말 출력
show_help() {
    echo "비디오 렌더링 스크립트"
    echo ""
    echo "사용법:"
    echo "  $0 <project_name> [output_file]"
    echo ""
    echo "예시:"
    echo "  $0 test                    # test 프로젝트를 기본 이름으로 렌더링"
    echo "  $0 test my_video.mp4       # test 프로젝트를 my_video.mp4로 렌더링"
    echo ""
    echo "옵션:"
    echo "  -h, --help                 이 도움말 출력"
    echo "  -l, --list                 사용 가능한 프로젝트 목록 출력"
    echo ""
}

# 프로젝트 목록 출력
list_projects() {
    echo "사용 가능한 프로젝트:"
    if [ -d "$PROJECTS_DIR" ]; then
        for project in "$PROJECTS_DIR"/*; do
            if [ -d "$project" ] && [ -f "$project/editing_state.json" ]; then
                project_name=$(basename "$project")
                echo "  - $project_name"
            fi
        done
    else
        echo "  프로젝트 디렉토리가 없습니다: $PROJECTS_DIR"
    fi
}

# 인자 처리
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    -l|--list)
        list_projects
        exit 0
        ;;
    "")
        echo "오류: 프로젝트 이름이 필요합니다."
        echo ""
        show_help
        exit 1
        ;;
esac

PROJECT_NAME="$1"
OUTPUT_FILE="$2"

# 프로젝트 디렉토리 확인
PROJECT_DIR="$PROJECTS_DIR/$PROJECT_NAME"
EDITING_STATE_FILE="$PROJECT_DIR/editing_state.json"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "오류: 프로젝트 디렉토리를 찾을 수 없습니다: $PROJECT_DIR"
    echo ""
    list_projects
    exit 1
fi

if [ ! -f "$EDITING_STATE_FILE" ]; then
    echo "오류: editing_state.json 파일을 찾을 수 없습니다: $EDITING_STATE_FILE"
    exit 1
fi

# Python 스크립트 확인
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "오류: 렌더링 스크립트를 찾을 수 없습니다: $SCRIPT_NAME"
    exit 1
fi

# 렌더링 실행
echo "프로젝트 렌더링 시작: $PROJECT_NAME"
echo "편집 상태 파일: $EDITING_STATE_FILE"

if [ -n "$OUTPUT_FILE" ]; then
    echo "출력 파일: $OUTPUT_FILE"
    python3 "$SCRIPT_NAME" "$EDITING_STATE_FILE" -o "$OUTPUT_FILE" -v
else
    echo "출력 파일: 자동 생성됨"
    python3 "$SCRIPT_NAME" "$EDITING_STATE_FILE" -v
fi

echo "렌더링 완료!" 