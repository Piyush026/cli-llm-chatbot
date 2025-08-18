import asyncio
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam
from rich import print

from tools import get_weather, search_wikipedia, calculate

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
HF_TOKEN = os.environ["HF_TOKEN"]
HF_ENDPOINT = os.environ["HF_ENDPOINT"]
HF_MODEL = os.environ["HF_MODEL"]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia and return a short summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression using Python math",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_wikipedia":search_wikipedia,
    "calculate":calculate
}

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
        if role == 'user':
            messages.append(ChatCompletionUserMessageParam(
                content=content, role="user"
            ))
        elif role == 'assistant':
            messages.append(ChatCompletionAssistantMessageParam(content=content, role="assistant"))

    return messages


async def call_llm_chat(client, messages, model):
    try:
        response = client.chat.completions.create(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=1,
            top_p=0.5,
            model=model
        )
        return response.choices[0].message
    except Exception as e:
        return {"content": f"(error calling LLM) {str(e)}"}


async def execute_tools(tool_calls, messages):
    results = []
    async def _run_tool(tool_call):
        func_name = tool_call.function.name
        if func_name in TOOL_FUNCTIONS:
            args = json.loads(tool_call.function.arguments)
            result = await TOOL_FUNCTIONS[func_name](**args)  # Assume async tools; use asyncio.gather for true parallel
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,  # Required for OpenAI API
                "name": func_name,
                "content": str(result)
            })
            print(messages)
            results.append(result)
        else:
            results.append(f"Unknown tool: {func_name}")
    return await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])

async def get_response(system_message, context):
    try:
        client = create_client(LLM_PROVIDER)
        messages = prepare_messages(system_message, context)
        model = HF_MODEL if LLM_PROVIDER == "huggingface" else OPENAI_MODEL
        max_iterations = 5  # Prevent infinite loops
        for _ in range(max_iterations):
            msg = await call_llm_chat(client, messages, model)
            messages.append(msg)  # Add assistant's message

            if not msg.tool_calls:
                return {"content": msg.content}  # Done if no tools called

            # Execute tools (supports multiple/parallel)
            await execute_tools(msg.tool_calls, messages)
        return {"content": "(Max iterations reached; partial response)"}
    except Exception as e:
        return {"content": f"(error calling LLM) {str(e)}"}
