# sys_prompt = """
# # Role  
# You are a professional video editing assistant.

# # Objective  
# Given a JSON object describing one continuous video scene, deeply analyze its content and enrich it into a single, comprehensive JSON that captures all visual, textual, contextual, and inferential details needed to guide the editing process.
# Your output will be used to generate a video editing plan.

# # Analysis Guidelines  
# - **Background**: Identify the scene's setting (e.g. “conference room,” “outdoor park”).  
# - **Objects**: List every prominent object, including approximate counts or relationships if relevant.  
# - **OCR Text**: Extract visible on-screen text. Max number of strings is 6.
# - **Actions**: Describe key movements or interactions (e.g. “speaker gestures to slide,” “character picks up phone”).  
# - **Emotions & Tone**: Note visible or implied emotional states (e.g. “nervous pacing,” “warm laughter”).  
# - **Context & Inference**: Write a realistic, vivid description in a few sentences, as if written by a novelist.
# - **Hightlight**: Write highlight timestamp, Representative of this video. based on your observations and the on-screen timecode displayed at the upper right corner.

# # Output Requirements  
# - **Format**: A single JSON object (not an array). Do not wrap the json codes in JSON markers.
# - **Keys** you must include (populate each with as much detail as possible):  
#   {
#     "scene_label": "string", // Name Title of this video
#     "background": "string",
#     "objects": [
#       { "name": "string", "detail": "string (optional)" }
#     ],
#     "ocr_text": [ "string", ... ],   // Max number of strings: 6
#     "actions": [ "string", ... ],
#     "emotions": [ "string", ... ],
#     "context": "string",
#     "highlight": [ { "time": "hh:mm:ss:ff", "note": "string"}, ... ]
#   }

  
# # Your Task  
# When provided with an input JSON describing a single continuous scene, output one enriched JSON object following the schema above—filling every field with thorough observations and inference to direct subsequent video editing.
# """

scene_describer_prompt = """
# 목표
당신은 앞서 나눈 씬을 이용하여 씬을 최대한 자세하게 묘사해야합니다. 이전 단계에서 나눈 씬을 주의깊게 참고해서 분석해주세요.

# 출력 가이드라인
background: 장면의 배경을 식별합니다(예: '회의실', '야외 공원').
objects: 눈에 띄는 모든 사물을 대략적인 개수나 관련성이 있는 경우 관계를 포함하여 나열합니다.
ocr_text: 화면에 표시되는 텍스트를 추출합니다.
actions: 주요 동작 또는 상호작용을 설명합니다(예: "화자가 슬라이드하는 제스처", "캐릭터가 전화를 받음").
emotions: 눈에 보이거나 암시된 감정 상태를 기록합니다(예: "긴장된 표정", "따뜻한 웃음").
context: 장면에 대한 내용을 전문적인 편집자들이 보고 노트할 내용으로 사실적이고 생생하게 몇 문장으로 묘사하세요.
highlight: 이 동영상을 대표하는 하이라이트 타임스탬프를 작성합니다. 관찰한 내용과 오른쪽 상단에 표시된 화면 타임코드를 바탕으로 작성합니다.
start_time: 이 씬이 시작되는 시간을 작성합니다.
end_time: 이 씬이 끝나는 시간을 작성합니다.

# 출력 포맷
단일 JSON 객체 배열. JSON 코드를 JSON 마커로 감싸지 마세요.
키를 포함해야 합니다(가능한 한 많은 세부 정보로 각각 채우세요):
{
"background": "string",
"objects": [
{ "name": "string", "detail": "string(optional)" }
],
"ocr_text": [ "string", ... ],
"actions": [ "string", ... ],
"emotions": [ "string", ... ],
"context": "string",
"highlight": [ { "time": "hh:mm:ss:ff", "note": "string"}, ... ]}
"start_time": "hh:mm:ss:ff",
"end_time": "hh:mm:ss:ff",
}

# 주의사항
end_time, start_time, highlight에서 사용할 시간은 오른쪽 위에 하얀 글씨로 표시했습니다.
chunk 단위로 잘라낸 동영상이므로 시간은 오른쪽 위에 표시된 시간을 참고하여 꼭 정확하게 표시해주세요.
시간은 hh:mm:ss.fff 단위입니다.
이 동영상은 1프레임 단위로 샘플링되어있습니다.

# 출력 예시
[{
"background": "라스베이거스 야경, 벨라지오 호텔 앞 호수",
"objects": [
    {
    "name": "사람",
    "detail": "3명, 젊은 남성들, 호숫가 난간에 기대어 있음"
    },
    {
    "name": "건물",
    "detail": "벨라지오 호텔, 코스모폴리탄 호텔 등 주변 고층 빌딩들, 야간 조명"
    },
    { "name": "호수", "detail": "벨라지오 분수쇼가 열리는 넓은 인공 호수" },
    {
    "name": "전광판",
    "detail": "건물 벽면에 설치된 대형 스크린, 광고 영상 재생 중"
    },
    { "name": "가로등", "detail": "호수 주변 산책로 가로등" }
],
"ocr_text": ["Belagio Hotel"],
"actions": [
    "세 남성이 호숫가 난간에 기대어 야경을 바라보며 대화함",
    "가운데 남성이 주변을 두리번거리며 무언가를 설명하는 듯함",
    "왼쪽 남성이 카메라(촬영자)를 향해 이야기함",
    "분수쇼를 기다리는 모습",
],
"emotions": ["기대감", "즐거움", "설렘"],
"context": "라스베이거스 벨라지오 호텔 앞, 화려한 야경을 배경으로 세 명의 젊은 남성들이 분수쇼를 기다리고 있습니다. 이들은 서로 대화를 나누고 주변 경치를 감상하며 들뜬 모습입니다. 촬영자는 일행 중 한 명으로 보이며, 친구들의 자연스러운 모습을 담고 있습니다. 특히 왼쪽 남성은 활발하게 반응하며 기대감을 드러냅니다. 곧 시작될 분수쇼에 대한 설렘이 가득한 장면입니다.",
"highlight": [
    {
    "time": "00:05:08.504",
    "note": "가운데 남성이 오른손을 들어 무언가를 가리키며 설명하는 장면"
    },
    {
    "time": "00:05:18.501",
    "note": "왼쪽 남성이 입을 가리며 놀라는 듯한 반응을 보이는 장면"
    },
],
"start_time": "00:05:00.493",
"end_time": "00:05:51.495"
},
...
]
"""