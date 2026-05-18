"""Backend for Anthropic API."""

import logging
import time
import os
import threading
import itertools

from .backend_utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, select_values
import anthropic
from aide.utils.config import Config

logger = logging.getLogger("aide")

# ---------------------------------------------------------
# 并发控制与多 Key 轮询池
# ---------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 5
_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

_client_iterator = None
_client_pool_lock = threading.Lock()

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)
ANTHROPIC_MODEL_ALIASES = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    
    # 场景 1：如果你在配置里写的是一个简单的名字，想让它在底层自动转换成 AWS Bedrock 格式的代理名：
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    
    # 场景 2：如果你在配置里写的就是 "us.anthropic.claude-sonnet-4-6" 以便通过 init.py 的路由，
    # 但你的代理服务器实际上只认官方的标准模型名，那就把它映射回官方名称：
    # "us.anthropic.claude-sonnet-4-6": "claude-3-5-sonnet-20241022",
}
def _setup_anthropic_client(cfg: Config | None = None):
    global _client_iterator
    if _client_iterator is not None:
        return
        
    # 尝试从 cfg 或环境变量读取 Base URL
    base_url = "https://www.litellm.org/"

    # 尝试读取 API Keys (支持逗号分隔的多 Key)
    api_keys_str = ""
    if cfg is not None and getattr(cfg, "anthropic_api_key", None):
        api_keys_str = cfg.anthropic_api_key
    if not api_keys_str:
        api_keys_str = os.getenv("ANTHROPIC_API_KEYS", os.getenv("ANTHROPIC_API_KEY", "sk-nFKLUhsGdMyhQfOrgAvofg , sk-eVOR3ItJVak7FS9iFPvluA"))
        
    # 清理并过滤掉空字符串
    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    
    if not api_keys:
        raise ValueError("No Anthropic API keys found. Please set ANTHROPIC_API_KEYS environment variable or pass via cfg.")

    logger.info(f"Initializing Anthropic client pool with {len(api_keys)} keys.")

    # 为每一个 Key 创建独立的 Anthropic Client 实例
    client_pool = []
    for key in api_keys:
        client_kwargs = {"max_retries": 0, "api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        client = anthropic.Anthropic(**client_kwargs)
        client_pool.append(client)

    # 创建无限循环迭代器
    _client_iterator = itertools.cycle(client_pool)

def get_next_client() -> anthropic.Anthropic:
    """线程安全地获取下一个轮询的客户端实例"""
    with _client_pool_lock:
        return next(_client_iterator)

def create_message_with_rotation(**kwargs):
    """
    代理函数：每次被调用时都会获取一个新的 Client。
    配合 backoff_create 使用时，如果遇到 429 重试，会自动切换到下一个 Key！
    """
    client = get_next_client()
    return client.messages.create(**kwargs)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query Anthropic's API, optionally with tool use (Anthropic's equivalent to function calling).
    """
    # 确保客户端池已初始化
    _setup_anthropic_client(cfg)

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 16384  # default for Claude models

    model_name = filtered_kwargs.get("model", "us.anthropic.claude-sonnet-4-6")
    logger.debug(f"Anthropic query called with model='{model_name}'")

    if model_name in ANTHROPIC_MODEL_ALIASES:
        model_name = ANTHROPIC_MODEL_ALIASES[model_name]
        filtered_kwargs["model"] = model_name
        logger.debug(f"Using aliased model name: {model_name}")

    if func_spec is not None and func_spec.name == "submit_review":
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        # Force tool use
        filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict

    # Anthropic doesn't allow not having user messages
    # if we only have system msg -> use it as user msg
# ---------------------------------------------------------
    # 【核心修复】：拦截并剥离 messages 数组里的 system 角色
    # ---------------------------------------------------------
    raw_messages = opt_messages_to_list(None, user_message)
    
    anthropic_messages = []
    system_prompts = []
    
    # 1. 收集原本作为单独参数传进来的 system_message
    if system_message:
        system_prompts.append(system_message)
        
    # 2. 遍历框架传来的消息数组，把混入其中的 system 强行剔除并收集
    for msg in raw_messages:
        if msg.get("role") == "system":
            if msg.get("content"):
                system_prompts.append(msg["content"])
        else:
            anthropic_messages.append(msg)
            
    # 3. 应对极端情况：Anthropic 强制要求至少有一条 user message
    # 如果全都是 system 没有 user，就把最后一段 system 当做 user 发送
    if not anthropic_messages and system_prompts:
        anthropic_messages.append({"role": "user", "content": system_prompts.pop()})
        
    # 4. 把收集到的 system 组合成顶层参数 (Anthropic 只认顶层的 system)
    if system_prompts:
        filtered_kwargs["system"] = "\n\n".join(system_prompts)

    # 重新赋值给真正的 messages 变量，保证里面是干净的（只有 user 和 assistant）
    messages = anthropic_messages

    logger.info(f"Anthropic API request: system={system_message}, user={user_message}")

    message = None
    t0 = time.time()
    
    try:
        logger.debug(f"Waiting for semaphore... Active threads: {MAX_CONCURRENT_REQUESTS - _concurrency_semaphore._value}")
        with _concurrency_semaphore:
            # 核心修改点：使用轮询代理函数包裹 messages.create
            message = backoff_create(
                create_message_with_rotation,
                ANTHROPIC_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
    except Exception as e:
        logger.error(f"Anthropic query failed: {str(e)}")
        raise
        
    req_time = time.time() - t0

    # Handle tool calls if present
    if (
        func_spec is not None
        and "tools" in filtered_kwargs
        and len(message.content) > 0
        and message.content[0].type == "tool_use"
    ):
        block = message.content[0]  # This is a "ToolUseBlock"
        assert (
            block.name == func_spec.name
        ), f"Function name mismatch: expected {func_spec.name}, got {block.name}"
        output = block.input  # Anthropic calls the parameters "input"
    else:
        # For non-tool responses, ensure we have text content
        assert len(message.content) == 1, "Expected single content item"
        assert (
            message.content[0].type == "text"
        ), f"Expected text response, got {message.content[0].type}"
        output = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
        "model": message.model,
    }

    logger.info(
        f"Anthropic API call completed - {message.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"Anthropic API response: {output}")

    return output, req_time, in_tokens, out_tokens, info