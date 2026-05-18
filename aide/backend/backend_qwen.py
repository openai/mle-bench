"""Backend for OpenAI API."""

import json
import logging
import re
import time
import os
import threading
from aide.backend.backend_utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai
from aide.utils.config import Config

logger = logging.getLogger("aide")
_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

_client_lock = threading.Lock()  # 2. 定义全局锁

# 3. 移除 @once，改用手动锁管理
def _get_openai_client(cfg: Config) -> openai.OpenAI:
    global _client
    if _client is None:  # 第一层检查：避免不必要的加锁开销
        with _client_lock:  # 加锁：确保只有一个线程能进入初始化逻辑
            if _client is None:  # 第二层检查：确认在等待锁的过程中没有被其他线程初始化
                logger.info("Initializing OpenAI client for the first time...")
                _client = openai.OpenAI(
                    base_url=cfg.agent.cheap.base_url,
                    api_key=cfg.agent.cheap.api_key,
                    max_retries=0,
                )
    return _client
    
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
    # 使用 chr(96) 动态生成连续的三个反引号，避免直接在代码里写出导致 Canvas 解析中断
    backticks = chr(96) * 3
    pattern = rf"{backticks}(?:json)?(.*?){backticks}"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
        # 如果没有 markdown 代码块，直接返回清理两端空格的内容
    return content.strip()

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg:Config | None = None, 
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = _get_openai_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # 有些模型取消注释下方这行能够强制其调用 function，可以根据需要打开
        # filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    message_print = messages[0]["content"]
    print(f"\033[31m{message_print}\033[0m")
    
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
        print(f"\033[32m{output}\033[0m")
    else:
        # 修改点：增加对 Tool Call 失败的降级处理
        if choice.message.tool_calls:
            assert (
                choice.message.tool_calls[0].function.name == func_spec.name
            ), "Function name mismatch"
            raw_args = choice.message.tool_calls[0].function.arguments
        else:
            # 降级：模型把 JSON 放到了普通的 message.content 里面
            content = choice.message.content
            if content:
                logger.warning(f"tool_calls is empty, fallback to extracting JSON from content.")
                raw_args = extract_json_from_content(content)
            else:
                # 如果两者都为空，这才抛出异常
                raise AssertionError(f"Both tool_calls and content are empty. Response: {choice.message}")
        fixed_args = fix_json_string(raw_args)

        # 修改点：真正启用 safe_json_loads，彻底避免报错中断
        output = safe_json_loads(fixed_args)
        
        # 打印一下降级日志，方便你调试
        if "raw_output" in output:
            logger.error(f"Fallback dict created due to JSONDecodeError. Raw string: {fixed_args}")

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info