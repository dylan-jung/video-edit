import os

import httpx
from openai import OpenAI

cookies = httpx.Cookies()
cookies.set("code-server-session",
            """%24argon2id%24v%3D19%24m%3D65536%2Ct%3D3%2Cp%3D4%24PH%2B%2FvHn4uG3JWvs5wWFSWQ%24Sl2b696Jv%2FO528n5NVQlakXBkt5MTfV8ODf8HiZAww0""",
            domain=".1a40432.tunnel.myubai.uos.ac.kr",
            path="/")

httpx_client = httpx.Client(
    cookies=cookies,
)

client = OpenAI(
    base_url=os.getenv("VISION_API_URL", "http://localhost:8001/v1"),
    api_key=os.getenv("VISION_API_KEY", "secretT"),
    http_client=httpx_client,
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": "ping"}],
    max_tokens=2,
)

print(resp.choices[0].message.content)
