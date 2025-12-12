from typing import Dict, Any, List, Union

def scene_to_string(scene: Dict[str, Any]) -> str:
    """
    Convert scene dictionary to string for embedding.
    """
    parts = []
    
    # Background
    if scene.get('background'):
        parts.append(f"배경: {scene['background']}")
    
    # Objects
    objects = scene.get('objects', [])
    if objects:
        descriptions = []
        for obj in objects:
            name = obj.get('name', '')
            detail = obj.get('detail', '')
            if name and detail:
                descriptions.append(f"{name} ({detail})")
            elif name:
                descriptions.append(name)
        if descriptions:
            parts.append(f"객체: {', '.join(descriptions)}")
    
    # OCR
    if scene.get('ocr_text'):
        parts.append(f"텍스트: {', '.join(scene['ocr_text'])}")
    
    # Actions
    if scene.get('actions'):
        parts.append(f"행동: {', '.join(scene['actions'])}")
    
    # Emotions
    if scene.get('emotions'):
        parts.append(f"감정: {', '.join(scene['emotions'])}")
    
    # Context
    if scene.get('context'):
        parts.append(f"상황: {scene['context']}")
    
    # Highlights
    highlights = scene.get('highlight', [])
    if highlights:
        descs = []
        for h in highlights:
            time = h.get('time', '')
            note = h.get('note', '')
            if time and note:
                descs.append(f"{time} - {note}")
        if descs:
            parts.append(f"하이라이트: {'; '.join(descs)}")
    
    # Time
    if scene.get('start_time') and scene.get('end_time'):
        parts.append(f"시간: {scene['start_time']} ~ {scene['end_time']}")
    
    return ' | '.join(parts)


def speech_to_string(chunk: Dict[str, Any]) -> str:
    """
    Convert speech chunk dictionary to string for semantic search embedding.
    """
    parts = []
    
    if chunk.get('summary'):
        parts.append(f"요약: {chunk['summary']}")
    
    if chunk.get('keywords'):
        parts.append(f"키워드: {', '.join(chunk['keywords'])}")
    
    if chunk.get('topics'):
        parts.append(f"주제: {', '.join(chunk['topics'])}")
    
    if chunk.get('text'):
        text = chunk['text']
        if isinstance(text, list):
            text = ' '.join(text)
        parts.append(f"내용: {text}")
    
    if chunk.get('context'):
        parts.append(f"맥락: {chunk['context']}")
    
    if chunk.get('sentiment'):
        parts.append(f"감정: {chunk['sentiment']}")
        
    if chunk.get('importance'):
        parts.append(f"중요도: {chunk['importance']}")
    
    return '. '.join(parts)
