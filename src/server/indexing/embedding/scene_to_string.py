def scene_to_string(scene: dict) -> str:
    """
    장면 정보를 하나의 문자열로 변환
    
    Args:
        scene: 장면 정보 딕셔너리
        
    Returns:
        장면 정보가 포함된 문자열
    """
    parts = []
    
    # 배경 정보
    background = scene.get('background', '')
    if background:
        parts.append(f"배경: {background}")
    
    # 객체 정보
    objects = scene.get('objects', [])
    if objects:
        object_descriptions = []
        for obj in objects:
            name = obj.get('name', '')
            detail = obj.get('detail', '')
            if name and detail:
                object_descriptions.append(f"{name} ({detail})")
            elif name:
                object_descriptions.append(name)
        if object_descriptions:
            parts.append(f"객체: {', '.join(object_descriptions)}")
    
    # OCR 텍스트
    ocr_text = scene.get('ocr_text', [])
    if ocr_text:
        parts.append(f"텍스트: {', '.join(ocr_text)}")
    
    # 행동/동작
    actions = scene.get('actions', [])
    if actions:
        parts.append(f"행동: {', '.join(actions)}")
    
    # 감정
    emotions = scene.get('emotions', [])
    if emotions:
        parts.append(f"감정: {', '.join(emotions)}")
    
    # 컨텍스트
    context = scene.get('context', '')
    if context:
        parts.append(f"상황: {context}")
    
    # 하이라이트
    highlights = scene.get('highlight', [])
    if highlights:
        highlight_descriptions = []
        for highlight in highlights:
            time = highlight.get('time', '')
            note = highlight.get('note', '')
            if time and note:
                highlight_descriptions.append(f"{time} - {note}")
        if highlight_descriptions:
            parts.append(f"하이라이트: {'; '.join(highlight_descriptions)}")
    
    # 시간 정보
    start_time = scene.get('start_time', '')
    end_time = scene.get('end_time', '')
    if start_time and end_time:
        parts.append(f"시간: {start_time} ~ {end_time}")
    
    return ' | '.join(parts)
