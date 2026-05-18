import logging
import traceback
from dataclasses import dataclass
from typing import Callable

import jsonschema
from dataclasses_json import DataClassJsonMixin
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

logger = logging.getLogger("aide")


def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception] | tuple, *args, **kwargs
):
    """
    带指数退避重试的 API 调用封装。
    使用 tenacity 拦截抛出的报错并自动重试。
    """
    # Tenacity 的 retry_if_exception_type 需要接收 tuple 类型的异常组
    exceptions_tuple = tuple(retry_exceptions) if isinstance(retry_exceptions, list) else retry_exceptions

    @retry(
        wait=wait_random_exponential(min=2, max=10),  # 随机等待 2~30 秒，完美错峰
        stop=stop_after_attempt(5),                   # 最多尝试 8 次
        retry=retry_if_exception_type(exceptions_tuple), # 仅针对指定的报错进行重试
        reraise=True                                  # 重试次数耗尽后抛出最后一次的异常
    )
    def _execute_with_retry():
        # 这里必须直接 return，让抛出的异常暴露给 @retry 装饰器
        return create_fn(*args, **kwargs)

    # 在最外层做兜底，兼容原有 aide 框架“失败则返回 False”的设计
    try:
        return _execute_with_retry()
    except exceptions_tuple as e:
        logger.error(f"Tenacity exhausted all retries! Last backoff exception: {e}")
        print(traceback.format_exc())
        return False


def opt_messages_to_list(
    system_message: str | None,
    user_message: str | None,
) -> list[dict[str, str]]:
    messages = []
    messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Anthropic-compatible tool dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Anthropic tool_choice dict."""
        return {
            "type": "tool",
            "name": self.name,
        }

    def parse_tool_call(self, tool_call) -> dict:
        """Parse an OpenAI tool_call object and return the arguments as a dict.

        Args:
            tool_call: OpenAI ChatCompletionMessageToolCall object.

        Returns:
            Parsed arguments dict.

        Raises:
            ValueError: If function name doesn't match or JSON parsing fails.
        """
        import json as _json

        func_name = getattr(tool_call.function, "name", None)
        if func_name and func_name != self.name:
            raise ValueError(
                f"Function name mismatch: expected '{self.name}', got '{func_name}'"
            )

        raw_args = getattr(tool_call.function, "arguments", "{}")
        if isinstance(raw_args, dict):
            return raw_args

        return _json.loads(raw_args)