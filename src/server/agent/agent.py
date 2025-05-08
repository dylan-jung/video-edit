import os

from dotenv import load_dotenv
from prompt import system_prompt, user_prompt
from qwen_agent.agents import Assistant
from tools import *

llm_cfg = {
    "model": os.getenv('AGENT_MODEL'),
    'model_server': os.getenv('AGENT_MODEL_SERVER'),
    # 'api_key': os.getenv('API_KEY'),
}

bot = Assistant(
    llm=llm_cfg,
    function_list=["code_interpreter", 'read_overview', 'read_video', 'write_project'],
    system_message=system_prompt,
)

history = []
while True:
    q = input("\nUser ▶ ")
    history.append({"role": "user", "content": q})
    for chunk in bot.run(messages=history, stream=False):
        pass
    print("Assistant Reasoning ▶", chunk[-2]["reasoning_content"])
    print("Assistant ▶", chunk[-1]["content"])
    history.extend(chunk)
