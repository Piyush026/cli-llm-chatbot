import os
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

async def call_openai_chat(messages):
    # token = ""
    # endpoint = "https://models.inference.ai.azure.com"
    # model_name = "gpt-4.1"
    client = OpenAI(
        base_url=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    )

    response = client.chat.completions.create(
        messages=[
        ChatCompletionSystemMessageParam(content="Expert in everything special in physiology",role="system",
                                         name="Fedrick"),
        ChatCompletionUserMessageParam(content=messages, role="user"),

    ],
        temperature=1,
        top_p=0.5,
        model=OPENAI_MODEL
    )
    content = response.choices[0].message.content
    return {"content": content}

async def get_response(messages):
    try:
        return await call_openai_chat(messages)
    except Exception as e:
        # graceful fallback on network / API errors
        return {"content": f"(error calling LLM) {str(e)}"}
