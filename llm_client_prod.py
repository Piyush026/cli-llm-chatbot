# llm_client.py
import json
import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from pydantic_settings import BaseSettings

from pydantic import Field, ValidationError, validator, field_validator


# ----------------------------------------------------------------------
# 1️⃣ Configuration
# ----------------------------------------------------------------------
class Provider(str, Enum):
    """Supported LLM back‑ends."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMSettings(BaseSettings):
    """All configuration comes from environment variables (or an optional .env file)."""

    # Provider selection -------------------------------------------------
    llm_provider: Provider = Field(..., env="LLM_PROVIDER")

    # OpenAI credentials ------------------------------------------------
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_endpoint: Optional[str] = Field(default=None, env="OPENAI_ENDPOINT")

    # HuggingFace credentials -------------------------------------------
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    hf_endpoint: Optional[str] = Field(default=None, env="HF_ENDPOINT")
    hf_model: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct", env="HF_MODEL")

    # Generation defaults ------------------------------------------------
    temperature: float = Field(default=1.0, env="TEMP")
    top_p: float = Field(default=0.5, env="TOP_P")
    max_tokens: Optional[int] = Field(default=None, env="MAX_TOKENS")

    # Operational knobs --------------------------------------------------
    retry_attempts: int = Field(default=3, env="RETRY_ATTEMPTS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    weather_api_key: Optional[str] = Field(default=None, env="WEATHER_API_KEY")
    # ------------------------------------------------------------------
    @field_validator("openai_api_key", "hf_token", mode="before")
    @classmethod
    def _ensure_token_present(cls, v: Optional[str], info):
        # `info` gives us the field name and the values that have already been parsed
        provider: Provider = info.data.get("llm_provider")
        field_name = info.field_name

        if provider == Provider.OPENAI and field_name == "openai_api_key" and not v:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        if provider == Provider.HUGGINGFACE and field_name == "hf_token" and not v:
            raise ValueError("HF_TOKEN must be set when LLM_PROVIDER=huggingface")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create a single validated settings instance at import time
try:
    settings = LLMSettings()
except ValidationError as exc:
    print(repr(exc.errors()))
    raise SystemExit(f"❌ Invalid configuration – {exc}")

# ----------------------------------------------------------------------
# 2️⃣ Logging
# ----------------------------------------------------------------------
log = logging.getLogger("llm")
log.setLevel(settings.log_level.upper())
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
log.addHandler(handler)

# ----------------------------------------------------------------------
# 3️⃣ Tool registry (plug‑and‑play for any LLM function calling)
# ----------------------------------------------------------------------
class ToolRegistry:
    """Maps a tool name – as defined in the JSON schema – to an async callable."""

    _registry: Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator used on async functions to add them to the registry."""
        def inner(func: Callable[[Dict[str, Any]], Awaitable[str]]):
            cls._registry[name] = func
            return func
        return inner

    @classmethod
    async def call(cls, name: str, arguments: Dict[str, Any]) -> str:
        if not isinstance(name, str):
            raise TypeError(f"Tool name must be a string, got {type(name).__name__}")

        if name not in cls._registry:
            raise ValueError(f"Tool '{name}' is not registered")

        log.debug("🔧 Calling tool %s with %s", name, arguments)
        return await cls._registry[name](arguments)


# ----------------------------------------------------------------------
# 4️⃣ Concrete tool – Weather (cached for speed)
# ----------------------------------------------------------------------
@ToolRegistry.register("get_weather")
# @lru_cache(maxsize=128)  # cheap in‑memory LRU – works well for a few dozen distinct cities
async def get_weather_tool(args: Dict[str, Any]) -> str:
    """
    Thin wrapper around the project's ``tools.get_weather`` helper.
    ``tools.get_weather`` must be an ``async`` function returning a plain string.
    """
    from tools import get_weather  # Imported lazily to keep the registry import‑free

    city = args.get("city")
    if not city:
        raise ValueError("Missing required argument 'city'")
    result = await get_weather(city)
    return str(result)

# ----------------------------------------------------------------------
# 5️⃣ Abstract chat client – a tiny contract used by the factory
# ----------------------------------------------------------------------
class BaseChatClient:
    """All LLM back‑ends must implement ``generate``."""

    def __init__(self, temperature: float, top_p: float, max_tokens: Optional[int] = None):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
    ) -> "ChatResponse":
        raise NotImplementedError

# ----------------------------------------------------------------------
# 6️⃣ OpenAI (and compatible) implementation
# ----------------------------------------------------------------------
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class OpenAIChatClient(BaseChatClient):
    """Wraps ``openai.AsyncOpenAI`` – works for OpenAI *and* HuggingFace inference APIs."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
    ) -> "ChatResponse":
        """Calls the model with retry + exponential back‑off."""
        payload: Dict[str, Any] = {
            "model": settings.openai_model if settings.llm_provider == Provider.OPENAI else settings.hf_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        log.debug(
            "🧠 Sending request to %s – %d messages – tools=%s",
            self.client.base_url,
            len(messages),
            bool(tools),
        )
        result = await self.client.chat.completions.create(**payload)
        choice = result.choices[0]
        return ChatResponse(
            content=choice.message.content,
            raw=result,
            tool_calls=getattr(choice.message, "tool_calls", None),
        )

# ----------------------------------------------------------------------
# 7️⃣ Factory – lazily creates a singleton client based on the env
# ----------------------------------------------------------------------
class ChatClientFactory:
    _instance: Optional[BaseChatClient] = None

    @classmethod
    def get_client(cls) -> BaseChatClient:
        if cls._instance is not None:
            return cls._instance

        common_kwargs = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
        }

        if settings.llm_provider == Provider.OPENAI:
            log.info("🚀 Using OpenAI provider")
            cls._instance = OpenAIChatClient(
                api_key=settings.openai_api_key,
                base_url=settings.openai_endpoint,
                **common_kwargs,
            )
        elif settings.llm_provider == Provider.HUGGINGFACE:
            log.info("🚀 Using HuggingFace provider")
            cls._instance = OpenAIChatClient(
                api_key=settings.hf_token,
                base_url=settings.hf_endpoint,
                **common_kwargs,
            )
        else:
            raise RuntimeError(f"Unsupported provider: {settings.llm_provider}")

        return cls._instance

# ----------------------------------------------------------------------
# 8️⃣ Minimal response wrapper (slots keep the object tiny)
# ----------------------------------------------------------------------
class ChatResponse:
    __slots__ = ("content", "raw", "tool_calls")

    def __init__(self, content: Optional[str], raw: Any, tool_calls: Optional[List[Any]] = None):
        self.content = content
        self.raw = raw
        self.tool_calls = tool_calls

    def __repr__(self) -> str:
        return f"<ChatResponse content={self.content!r} tool_calls={len(self.tool_calls or [])}>"

# ----------------------------------------------------------------------
# 9️⃣ Helper: build the conversation payload
# ----------------------------------------------------------------------
def build_conversation(
    system_message: str,
    context: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Turn a list of ``{'role': ..., 'content': ...}`` dicts into the format
    expected by the OpenAI SDK. The explicit *system* prompt always sits
    at the top – this mirrors the original script.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_message}]
    for entry in context:
        role = entry.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role '{role}' in context")
        messages.append({"role": role, "content": entry.get("content", "")})
    return messages

# ----------------------------------------------------------------------
# 10️⃣ JSON schema for the weather tool (exposed to the model)
# ----------------------------------------------------------------------
def weather_tool_schema() -> Dict[str, Any]:
    """Schema fed to the model when tool calling is enabled."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "Name of the city"}},
                "required": ["city"],
            },
        },
    }

# ----------------------------------------------------------------------
# 11️⃣ Resolve tool calls returned by the model
# ----------------------------------------------------------------------
async def execute_tool_calls(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert each LLM tool call into a ``{'role':'tool', 'tool_call_id':..., 'name':..., 'content':...}``
    dict that can be appended to the conversation history.

    The function works with:
      * OpenAI < 1.0 (attribute‑style objects)
      * OpenAI ≥ 1.0 (TypedDict / Pydantic model)
    """
    responses: List[Dict[str, Any]] = []

    for tool_call in tool_calls:
        # --------------------------------------------------------------
        # 1️⃣ Normalise the tool‑call into a plain Python dict
        # --------------------------------------------------------------
        if isinstance(tool_call, dict):
            # New SDK – plain dict (TypedDict)
            func_dict = tool_call.get("function", {})
            tool_call_id = tool_call.get("id")
        else:   # pragma: no cover – old SDK path
            # Old SDK – model with attribute access
            func_dict = getattr(tool_call, "function", {})
            tool_call_id = getattr(tool_call, "id", None)

        # --------------------------------------------------------------
        # 2️⃣ Extract name (must be a string) and raw arguments
        # --------------------------------------------------------------
        fn_name = func_dict.get("name") if isinstance(func_dict, dict) else getattr(func_dict, "name", None)
        raw_args = func_dict.get("arguments") if isinstance(func_dict, dict) else getattr(func_dict, "arguments", "{}")

        if not isinstance(fn_name, str):
            log.warning(
                "Tool call did not contain a string name – skipping. Full payload: %s",
                tool_call,
            )
            continue

        # --------------------------------------------------------------
        # 3️⃣ Parse arguments – they may already be a dict (new SDK) or a JSON
        #    string (old SDK).  If they are malformed we fall back to an empty
        #    dict and log the problem.
        # --------------------------------------------------------------
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                log.error(
                    "Failed to decode JSON arguments for tool %s – %s",
                    fn_name,
                    exc,
                )
                arguments = {}
        else:
            # Already a dict / Mapping
            arguments = raw_args or {}

        # --------------------------------------------------------------
        # 4️⃣ Call the registered Python implementation.
        # --------------------------------------------------------------
        try:
            result = await ToolRegistry.call(fn_name, arguments)
        except Exception as exc:   # pylint: disable=broad-except
            # We never want a tool failure to crash the whole request.
            log.exception("Tool execution failed for %s", fn_name)
            result = f"Tool execution error ({fn_name}): {exc}"

        # --------------------------------------------------------------
        # 5️⃣ Build the tool‑response message **including the tool_call_id**.
        # --------------------------------------------------------------
        tool_message: Dict[str, Any] = {
            "role": "tool",
            "name": fn_name,
            "content": result,
        }
        # The API requires the ID – if we couldn't retrieve it we raise a warning
        # and fall back to the older 'id' key which the server will ignore.
        if tool_call_id:
            tool_message["tool_call_id"] = tool_call_id
        else:
            log.warning(
                "Tool call %s (%s) did not contain an 'id'. "
                "Appending tool message without tool_call_id may cause API errors.",
                fn_name,
                raw_args,
            )

        responses.append(tool_message)

    return responses
# ----------------------------------------------------------------------
# 12️⃣ Public entry point – used by a web server, bot, CLI, etc.
# ----------------------------------------------------------------------
async def get_response(system_message: str, context: List[Dict[str, str]]) -> ChatResponse:
    """
    1️⃣ Build the initial messages list.
    2️⃣ Send it to the configured LLM (with the weather tool schema enabled).
    3️⃣ If the model requested a tool, call it, append the result and ask the model
       to continue (second round‑trip).
    4️⃣ Return the final plain‑text answer.
    """
    messages = build_conversation(system_message, context)
    client = ChatClientFactory.get_client()

    # The weather tool is the only one we ship for now – add more items to the list
    # if you implement additional functions.
    tools = [weather_tool_schema()]

    # --------------------------------------------------------------
    # First pass – let the model decide whether it needs a tool.
    # --------------------------------------------------------------
    response = await client.generate(messages=messages, tools=tools, tool_choice="auto")

    # --------------------------------------------------------------
    # If the model asked for a tool, run it and do a second pass.
    # --------------------------------------------------------------
    if response.tool_calls:
        log.info("🛠️ Model requested %d tool call(s)", len(response.tool_calls))
        assistant_message = {
            "role": "assistant",
            "content": response.content,  # Usually None when tool_calls are present
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response.tool_calls
            ] if hasattr(response.tool_calls[0], 'id') else response.tool_calls  # Handle object or dict formats
        }
        messages.append(assistant_message)
        tool_messages = await execute_tool_calls(response.tool_calls)
        messages.extend(tool_messages)               # feed the tool output back to the LLM
        response = await client.generate(messages=messages)  # second pass – no tool schema needed

    final_content = response.content or ""
    return ChatResponse(content=final_content, raw=response.raw, tool_calls=response.tool_calls)

# ----------------------------------------------------------------------
# 13️⃣ Demo (run the file directly for a quick sanity check)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    DEMO_SYSTEM = "You are a helpful assistant that can retrieve weather data when needed."
    DEMO_CONTEXT = [
        {"role": "user", "content": "What’s the weather like in Paris right now?"}
    ]

    async def _run():
        reply = await get_response(DEMO_SYSTEM, DEMO_CONTEXT)
        print("🤖 Assistant:", reply.content)

    asyncio.run(_run())