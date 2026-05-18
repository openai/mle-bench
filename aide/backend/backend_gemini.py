"""Backend for Gemini API using OpenAI-compatible interface."""

import json
import logging
import os
import time
import re
import threading
import itertools

from .backend_utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from aide.utils.config import Config

logger = logging.getLogger("aide")

# ---------------------------------------------------------
# 并发控制与多 Key 轮询池
# ---------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 5
_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

_client_iterator = None
_client_pool_lock = threading.Lock()

GEMINI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def _setup_gemini_client(cfg: Config | None = None):
    global _client_iterator
    if _client_iterator is not None:
        return
        
    gemini_base_url = os.getenv("GEMINI_BASE_URL", "https://www.litellm.org/")
    
    # 从环境变量读取多个 Key，使用逗号分隔。
    # 示例: export GEMINI_API_KEYS="sk-key1,sk-key2,sk-key3"
    api_keys_env = os.getenv("GEMINI_API_KEYS", "sk-nFKLUhsGdMyhQfOrgAvofg , sk-eVOR3ItJVak7FS9iFPvluA")
    
    # 清理并过滤掉空字符串
    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    
    if not api_keys:
        raise ValueError("No API keys found. Please set GEMINI_API_KEYS environment variable.")

    logger.info(f"Initializing Gemini client pool with {len(api_keys)} keys.")

    # 为每一个 Key 创建一个独立的 OpenAI Client 实例
    client_pool = []
    for key in api_keys:
        client = openai.OpenAI(api_key=key, base_url=gemini_base_url, max_retries=0)
        client_pool.append(client)

    # 使用 itertools.cycle 创建一个无限循环的迭代器 (A -> B -> C -> A -> B ...)
    _client_iterator = itertools.cycle(client_pool)

def get_next_client() -> openai.OpenAI:
    """线程安全地获取下一个轮询的客户端实例"""
    _setup_gemini_client()
    with _client_pool_lock:
        return next(_client_iterator)

def create_completion_with_rotation(**kwargs):
    """
    代理函数：每次被调用时都会获取一个新的 Client。
    配合 backoff_create 使用时，如果遇到 429 重试，会自动切换到下一个 Key！
    """
    client = get_next_client()
    return client.chat.completions.create(**kwargs)


def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON, returning fallback dict: {s}")
        return {"raw_output": s}

def fix_json_string(s: str) -> str:
    """Fix common JSON parsing issues."""
    s = s.replace("\\'", "'")
    s = re.sub(r':\s*None\s*([,}])', r': null\1', s)
    return s

def extract_json_from_content(content: str) -> str:
    """Fallback: Extract JSON string from markdown content if tool_call fails."""
    if not content:
        return ""
    backticks = chr(96) * 3
    pattern = rf"{backticks}(?:json)?(.*?){backticks}"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    return content.strip()


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None, 
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the Gemini API via OpenAI-compatible interface, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    """
    # 确保客户端池已初始化
    _setup_gemini_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    messages = opt_messages_to_list(system_message, user_message)
    
    for msg in messages:
        if "content" not in msg or msg["content"] is None:
            msg["content"] = ""

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.info(f"Gemini API request: system={system_message}, user={user_message}")
    
    message_print = messages[0]["content"] if messages else ""
    if message_print:
        print(f"\033[31m{message_print[:200]}...\033[0m") 

    completion = None
    t0 = time.time()

    try:
        logger.debug(f"Waiting for semaphore... Active threads: {MAX_CONCURRENT_REQUESTS - _concurrency_semaphore._value}")
        with _concurrency_semaphore:
            # 修改点：将 _client.chat.completions.create 替换为动态轮询函数
            completion = backoff_create(
                create_completion_with_rotation,
                GEMINI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
    except openai.BadRequestError as e:
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning("Function calling not supported by this model. Falling back to plain text.")
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            with _concurrency_semaphore:
                # 修改点：降级重试同样使用动态轮询函数
                completion = backoff_create(
                    create_completion_with_rotation,
                    GEMINI_TIMEOUT_EXCEPTIONS,
                    messages=messages,
                    **filtered_kwargs,
                )
        else:
            raise

    req_time = time.time() - t0
    choice = completion.choices[0]

    if func_spec is None or "tools" not in filtered_kwargs:
        output = choice.message.content
        print(f"\033[32m{output[:200]}...\033[0m")
    else:
        tool_calls = getattr(choice.message, "tool_calls", None)
        raw_args = ""
        
        if tool_calls:
            first_call = tool_calls[0]
            if first_call.function.name != func_spec.name:
                logger.warning(f"Function name mismatch: expected {func_spec.name}, got {first_call.function.name}.")
            raw_args = first_call.function.arguments
        else:
            content = choice.message.content
            if content:
                logger.warning("tool_calls is empty, fallback to extracting JSON from content.")
                raw_args = extract_json_from_content(content)
            else:
                raise AssertionError(f"Both tool_calls and content are empty. Response: {choice.message}")
        
        fixed_args = fix_json_string(raw_args)
        output = safe_json_loads(fixed_args)
        
        if "raw_output" in output:
            logger.error(f"Fallback dict created due to JSONDecodeError. Raw string: {fixed_args}")

    in_tokens = getattr(completion.usage, "prompt_tokens", 0)
    out_tokens = getattr(completion.usage, "completion_tokens", 0)

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": getattr(completion, "model", "gemini-unknown"),
        "created": getattr(completion, "created", None),
    }

    logger.info(f"Gemini API completed - {info['model']} - {req_time:.2f}s - {in_tokens + out_tokens} tokens")

    return output, req_time, in_tokens, out_tokens, info