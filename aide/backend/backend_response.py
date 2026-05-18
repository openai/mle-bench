"""Backend for OpenAI API."""

import json

import logging

import re

import time

import os



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





@once

def _setup_openai_client(cfg:Config):

    global _client

    _client = openai.OpenAI(

        base_url=cfg.agent.feedback.base_url,

        api_key=cfg.agent.feedback.api_key,

        max_retries=3,

    )



def safe_json_loads(s: str) -> dict:

    try:

        return json.loads(s)

    except json.JSONDecodeError:

        # 如果解析失败，返回一个空 dict 或者包含原始字符串

        logger.warning(f"Failed to parse JSON, returning fallback dict: {s}")

        return {"raw_output": s}



def fix_json_string(s: str) -> str:

    """Fix common JSON parsing issues.

    

    This function fixes:

    1. Invalid \' escape sequences (replaces with ')

    2. Python None values (replaces with null)

    """

    # Replace invalid \' escape sequences with ' (single quotes don't need escaping in JSON)

    s = s.replace("\\'", "'")

    # Replace Python None with JSON null (must be a whole word to avoid false positives)

    import re

    # Replace None (as a JSON value) with null, but be careful with word boundaries

    # Pattern: "key": None (with optional whitespace)

    s = re.sub(r':\s*None\s*([,}])', r': null\1', s)

    return s



def query(

    system_message: str | None,

    user_message: str | None,

    func_spec: FunctionSpec | None = None,

    cfg:Config=None,

    **model_kwargs,

) -> tuple[OutputType, float, int, int, dict]:

    _setup_openai_client(cfg)

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore



    messages = opt_messages_to_list(system_message, user_message)



    if func_spec is not None:

        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]

        # force the model the use the function

        # filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()

    message_print = messages[0]["content"]

    print(f"\033[31m{message_print}\033[0m")

    filtered_kwargs.pop("convert_system_to_user", None)

    completion = backoff_create(

        _client.chat.completions.create,

        OPENAI_TIMEOUT_EXCEPTIONS,

        messages=messages,

        extra_body = {
        "enable_code_interpreter": False,
        # 代码解释器功能仅支持思考模式调用
        "enable_thinking": True,
    },
        stream =True,
        stream_options={"include_usage": True},

        **filtered_kwargs,

    )

    # 初始计数和计时
    t0 = time.time()
    full_content = ""
    full_reasoning = ""
    full_tool_args = ""
    in_tokens = 0
    out_tokens = 0
    info = {}
    
    # 假设 completion 是 backoff_create 返回的生成器
    if not completion:
        return None, 0, 0, 0, {"error": "Request failed"}
    
    for chunk in completion:
        # 1. 提取 Token 使用量 (通常在最后一个 chunk 或每个 chunk 都有)
        if hasattr(chunk, 'usage') and chunk.usage:
            in_tokens = chunk.usage.prompt_tokens
            out_tokens = chunk.usage.completion_tokens
    
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
    
        # 2. 收集思考过程 (Thinking)
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            full_reasoning += delta.reasoning_content
            # 可选：实时打印思考过程
            # print(f"\033[34m{delta.reasoning_content}\033[0m", end="", flush=True)
    
        # 3. 收集普通文本 (Content)
        if delta.content:
            full_content += delta.content
            print(delta.content, end="", flush=True)
    
        # 4. 收集函数调用 (Tool Calls)
        if delta.tool_calls:
            if delta.tool_calls[0].function.arguments:
                full_tool_args += delta.tool_calls[0].function.arguments
    
    print() # 打印换行
    req_time = time.time() - t0
    
    # --- 最终输出处理 ---
    
    if func_spec is None:
        output = full_content
    else:
        # 验证函数调用并解析 JSON
        assert full_tool_args, "function_call is empty in stream"
        fixed_args = fix_json_string(full_tool_args)
        try:
            output = json.loads(fixed_args)
        except json.JSONDecodeError:
            logger.error(f"Error decoding: {fixed_args}")
            raise
    
    # 填充 info 对象（包含思考过程等额外信息）
    info = {
        "reasoning": full_reasoning,
        "finish_reason": chunk.choices[0].finish_reason if chunk.choices else None
    }
    
    return output, req_time, in_tokens, out_tokens, info