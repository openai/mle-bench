"""Backend for OpenRouter API"""

import logging
import os
import time

from funcy import notnone, once, select_values
import openai

from .backend_utils import FunctionSpec, OutputType, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    if func_spec is not None:
        raise NotImplementedError(
            "We are not supporting function calling in OpenRouter for now."
        )

    # in case some backends dont support system roles, just convert everything to user
    messages = [
        {"role": "user", "content": message}
        for message in [system_message, user_message]
        if message
    ]

    logger.info(f"OpenRouter API request: system={system_message}, user={user_message}")

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    output = completion.choices[0].message.content

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    logger.info(
        f"OpenRouter API call completed - {completion.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"OpenRouter API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
