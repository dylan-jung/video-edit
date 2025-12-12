def speech_to_string(chunk: dict) -> str:
    """
    음성 청크로부터 시멘틱 검색용 텍스트 생성
    
    Args:
        chunk: 음성 청크 정보
        
    Returns:
        검색용 종합 텍스트
    """
    parts = []
    
    # 요약이 있으면 추가
    if chunk.get('summary'):
        parts.append(f"요약: {chunk['summary']}")
    
    # 키워드들 추가
    if chunk.get('keywords'):
        keywords_str = ', '.join(chunk['keywords'])
        parts.append(f"키워드: {keywords_str}")
    
    # 토픽들 추가
    if chunk.get('topics'):
        topics_str = ', '.join(chunk['topics'])
        parts.append(f"주제: {topics_str}")
    
    # 원본 텍스트 추가
    if chunk.get('text'):
        if isinstance(chunk['text'], list):
            text_str = ' '.join(chunk['text'])
        else:
            text_str = str(chunk['text'])
        parts.append(f"내용: {text_str}")
    
    # 컨텍스트 추가
    if chunk.get('context'):
        parts.append(f"맥락: {chunk['context']}")
    
    # 감정 정보 추가
    if chunk.get('sentiment'):
        parts.append(f"감정: {chunk['sentiment']}")
    
    # 중요도 정보 추가
    if chunk.get('importance'):
        parts.append(f"중요도: {chunk['importance']}")
    
    return '. '.join(parts)