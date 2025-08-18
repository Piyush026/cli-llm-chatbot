import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam
from rich import print

from tools import get_weather

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
HF_TOKEN = os.environ["HF_TOKEN"]
HF_ENDPOINT = os.environ["HF_ENDPOINT"]
HF_MODEL = os.environ["HF_MODEL"]


def create_client(provider):
    if provider == "huggingface":
        return OpenAI(base_url=HF_ENDPOINT, api_key=HF_TOKEN)
    else:
        return OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)


def prepare_messages(system_message, context):
    messages = [ChatCompletionSystemMessageParam(content=system_message, role="system", name="Fedrick")]
    for msg in context:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            messages.append(ChatCompletionSystemMessageParam(
                content=system_message,
                name="Fedrick",
                role="system"))
        elif role == 'user':
            messages.append(ChatCompletionUserMessageParam(
                content=content, role="user"
            ))
        elif role == 'assistant':
            messages.append(ChatCompletionAssistantMessageParam(content=content, role="assistant"))
        else:
            raise ValueError(f"Unsupported role: {role}")
    return messages


async def call_llm_chat(client, messages, model):
    try:
        response = client.chat.completions.create(
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "get current weather for the city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"]
                        }
                    }
                }
            ],
            tool_choice="auto",
            temperature=1,
            top_p=0.5,
            model=model
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "get_weather":
                    arg = json.loads(tool_call.function.arguments)
                    result = await get_weather(arg["city"])
                    messages.append({"role": "tool", "content": result})
                    return {"content": result}
        return {"content": msg.content}
    except Exception as e:
        return {"content": f"(error calling LLM) {str(e)}"}


async def get_response(system_message, context):
    try:
        client = create_client(LLM_PROVIDER)
        messages = prepare_messages(system_message, context)
        model = HF_MODEL if LLM_PROVIDER == "huggingface" else OPENAI_MODEL
        return await call_llm_chat(client, messages, model)
    except Exception as e:
        return {"content": f"(error calling LLM) {str(e)}"}
