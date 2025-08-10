import os
import aiohttp
from openai.types.beta.realtime.conversation_item_with_reference import Content
from rich import print
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
HF_TOKEN = os.environ["HF_TOKEN"]
HF_ENDPOINT = os.environ["HF_ENDPOINT"]
HF_MODEL = os.environ["HF_MODEL"]

async def call_openai_chat(messages,role):
    client = OpenAI(
        base_url=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    )

    response = client.chat.completions.create(
        messages=[
        ChatCompletionSystemMessageParam(content="Expert in everything special in physiology",role=role,
                                         name="Fedrick"),
        ChatCompletionUserMessageParam(content=messages, role="user")

    ],
        temperature=1,
        top_p=0.5,
        model=OPENAI_MODEL
    )
    content = response.choices[0].message.content
    return {"content": content}


async def call_huggingface_chat(messages,system_message,context):
    client = OpenAI(
        base_url=HF_ENDPOINT,
        api_key=HF_TOKEN,
    )
    messages = []
    try:
        for msg in context:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                messages.append(ChatCompletionSystemMessageParam(
                    content=system_message,
                    name="Fedrick",  # Keep the name as in your original code; can be made optional if needed
                role="system"))
            elif role == 'user':
                messages.append(ChatCompletionUserMessageParam(
                    content=content,role="user"
                ))
            elif role == 'assistant':
                messages.append(ChatCompletionAssistantMessageParam(
                    content=content,role="assistant"
                ))
            else:
                raise ValueError(f"Unsupported role: {role}")
        response = client.chat.completions.create(
            messages=messages,
            temperature=1,
            top_p=0.5,
            model=HF_MODEL
        )
        content = response.choices[0].message.content
        print(response)
        print(context)
        return {"content": content}
    except Exception as E:
        print(E)

async def get_response(messages,system_message,context):
    try:
        if LLM_PROVIDER == "huggingface":
            return await call_huggingface_chat(messages,system_message,context)
        return await call_openai_chat(messages,system_message)
    except Exception as e:
        # graceful fallback on network / API errors
        return {"content": f"(error calling LLM) {str(e)}"}
