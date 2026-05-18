"""Backend for OpenAI API (compatible with LiteLLM proxy, Azure, and OpenAI-compatible endpoints).

Key design decisions:
- Uses `response_format` with JSON Schema for structured output (more widely supported than tool calling).
- Falls back to tool calling if response_format is rejected (non-Azure only).
- Falls back to plain text + JSON extraction as last resort.
- Azure-compatible: automatically detects and drops unsupported params (tool_choice).
- API key rotation via round-robin pool for load balancing across multiple keys.

Fallback chain for structured output:
1. response_format (JSON Schema) → most widely supported
2. tool calling (tools + tool_choice) → skipped for Azure
3. plain text + JSON extraction → universal fallback
"""

import json
import logging
import os
import threading
import itertools
import time
import re
from typing import Any

from .backend_utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, select_values
import openai
from aide.utils.config import Config

logger = logging.getLogger("aide")

# ---------------------------------------------------------
# Concurrency control & API key rotation pool
# ---------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 5
_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

_client_iterator = None
_client_pool_lock = threading.Lock()

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# Azure/LiteLLM 不支持的参数列表
AZURE_UNSUPPORTED_PARAMS = {"tool_choice"}


def _is_azure_model(model: str | None, base_url: str | None = None) -> bool:
    """Detect if the request is targeting Azure.

    Azure (via LiteLLM proxy) doesn't support `tool_choice` parameter.
    We detect this by checking base_url for 'azure' or model name patterns.
    """
    if base_url and "azure" in base_url.lower():
        return True
    # Check if OPENAI_API_TYPE is set to azure
    api_type = os.getenv("OPENAI_API_TYPE", "").lower()
    if api_type == "azure":
        return True
    # Some Azure deployments use model names starting with 'azure/'
    if model and model.startswith("azure/"):
        return True
    return False


def _sanitize_kwargs_for_azure(kwargs: dict, model: str | None = None) -> dict:
    """Remove parameters unsupported by Azure models.

    Azure via LiteLLM doesn't support `tool_choice`. When detected,
    we drop it and rely on the model's default tool selection behavior.
    """
    sanitized = dict(kwargs)
    base_url = os.getenv("OPENAI_BASE_URL", "")

    if _is_azure_model(model, base_url):
        for param in AZURE_UNSUPPORTED_PARAMS:
            if param in sanitized:
                logger.warning(
                    f"Dropping unsupported param '{param}' for Azure model. "
                    f"Tool calling will use model default behavior."
                )
                del sanitized[param]

    return sanitized


def _should_strip_tool_params(err_msg: str) -> bool:
    """Check if error message indicates tool_choice/tools params should be stripped.

    This handles Azure and other providers that don't support these parameters.
    """
    indicators = [
        "tool_choice",
        "does not support parameters",
        "unsupportedparams",
        "unsupported params",
        "drop_params",
    ]
    err_lower = err_msg.lower()
    return any(ind in err_lower for ind in indicators)


def _setup_openai_client(cfg: Config | None = None):
    """Initialize the OpenAI client pool with round-robin API key rotation."""
    global _client_iterator
    if _client_iterator is not None:
        return

    base_url = os.getenv("OPENAI_BASE_URL", "https://www.litellm.org/")

    # Read multiple API keys from hardcoded string (comma-separated)
    api_keys_env = "sk-nFKLUhsGdMyhQfOrgAvofg , sk-eVOR3ItJVak7FS9iFPvluA"

    if not api_keys_env:
        raise ValueError("No API keys found. Please set OPENAI_API_KEYS or OPENAI_API_KEY.")

    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    if not api_keys:
        raise ValueError("No valid API keys after parsing.")

    logger.info(f"Initializing OpenAI client pool with {len(api_keys)} keys.")

    client_pool = []
    for key in api_keys:
        client = openai.OpenAI(api_key=key, base_url=base_url, max_retries=0)
        client_pool.append(client)

    _client_iterator = itertools.cycle(client_pool)


def get_next_client() -> openai.OpenAI:
    """Thread-safe round-robin client selection."""
    _setup_openai_client()
    with _client_pool_lock:
        return next(_client_iterator)


def create_completion_with_rotation(**kwargs):
    """Proxy that rotates client on each call. Works with backoff_create for 429 retry + key switch."""
    client = get_next_client()
    return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------
# Robust JSON parsing helpers
# ---------------------------------------------------------

def safe_json_loads(s: str) -> dict:
    """Parse JSON with fallback. Never raises."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parse failed: {e}. Returning raw_output fallback.")
        return {"raw_output": str(s)}


def fix_json_string(s: str) -> str:
    """Fix common JSON issues: Python None -> null, escaped quotes."""
    if not s:
        return s
    s = s.replace("\\'", "'")
    s = re.sub(r':\s*None\s*([,}])', r': null\1', s)
    return s


def extract_json_from_content(content: str) -> str:
    """Extract JSON from markdown code blocks or raw content."""
    if not content:
        return ""

    # Try markdown code block first: ```json ... ``` or ``` ... ```
    backticks = chr(96) * 3
    pattern = rf"{backticks}(?:json)?\s*\n?(.*?){backticks}"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to find JSON object directly (from first { to last })
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1].strip()

    return content.strip()


def _build_response_format(func_spec: FunctionSpec) -> dict:
    """Build OpenAI response_format dict for structured output (JSON Schema mode).

    This is preferred over tool calling because:
    1. More widely supported by LiteLLM proxies and OpenAI-compatible endpoints
    2. No tool_choice compatibility issues
    3. Model always returns JSON matching the schema
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": func_spec.name,
            "description": func_spec.description,
            "schema": func_spec.json_schema,
            "strict": True,
        }
    }


def _validate_and_fix_output(output: dict, func_spec: FunctionSpec) -> dict:
    """Validate output against schema and fill missing required fields with defaults."""
    schema = func_spec.json_schema
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})

    fixed = dict(output)
    for field in required_fields:
        if field not in fixed:
            prop = properties.get(field, {})
            field_type = prop.get("type", "string")

            # Provide sensible defaults based on type
            if field_type == "boolean":
                fixed[field] = False
            elif field_type == "number" or (isinstance(field_type, list) and "number" in field_type):
                fixed[field] = None
            elif field_type == "string":
                fixed[field] = ""
            elif field_type == "array":
                fixed[field] = []
            elif field_type == "object":
                fixed[field] = {}
            else:
                fixed[field] = None

            logger.warning(f"Missing required field '{field}' in LLM response. Using default: {fixed[field]}")

    return fixed


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI-compatible API with structured output support.

    Strategy for func_spec (structured output):
    1. Primary: Use `response_format` with JSON Schema (best compatibility)
    2. Fallback 1: Use tool calling (tools + tool_choice)
    3. Fallback 2: Plain text + JSON extraction from content

    Args:
        system_message: System prompt.
        user_message: User prompt.
        func_spec: Optional FunctionSpec for structured JSON output.
        cfg: Config object (used for client setup).
        **model_kwargs: Model parameters (temperature, model, etc.)

    Returns:
        (output, req_time, in_tokens, out_tokens, info)
        - output: str if func_spec is None, dict otherwise
    """
    _setup_openai_client(cfg)

    filtered_kwargs = select_values(notnone, model_kwargs)
    messages = opt_messages_to_list(system_message, user_message)

    # Track which structured output method we're using
    use_response_format = False
    use_tool_calling = False

    if func_spec is not None:
        # Strategy 1: Try response_format with JSON Schema first
        # This is more widely supported than tool calling across LiteLLM proxies
        filtered_kwargs["response_format"] = _build_response_format(func_spec)
        use_response_format = True

        # Also add a system hint about the expected JSON structure
        # This helps models that partially support response_format
        schema_hint = (
            f"\n\nIMPORTANT: You must respond with a valid JSON object matching this schema:\n"
            f"{json.dumps(func_spec.json_schema, indent=2, ensure_ascii=False)}"
        )
        # Append schema hint to system message
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += schema_hint
        elif messages:
            # If no system message, prepend to first user message
            messages[0]["content"] = schema_hint + "\n\n" + messages[0]["content"]

    t0 = time.time()
    completion = None
    model_name = filtered_kwargs.get("model")

    # ---------------------------------------------------------
    # API call with semaphore + backoff + key rotation
    # ---------------------------------------------------------
    try:
        logger.debug(
            f"Waiting for semaphore... Active requests: "
            f"{MAX_CONCURRENT_REQUESTS - _concurrency_semaphore._value}"
        )
        with _concurrency_semaphore:
            completion = backoff_create(
                create_completion_with_rotation,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
    except openai.BadRequestError as e:
        err_msg = str(e).lower()

        # If response_format is rejected, try tool calling
        if use_response_format and (
            "response_format" in err_msg
            or "json_schema" in err_msg
            or "unsupported" in err_msg
            or "invalid" in err_msg
            or "rejected" in err_msg
        ):
            logger.warning(
                f"response_format rejected ({err_msg[:150]}...). "
                f"Falling back to tool calling."
            )
            filtered_kwargs.pop("response_format", None)
            use_response_format = False

            # Try tool calling as fallback
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

            # Pre-check: sanitize for known Azure models
            filtered_kwargs = _sanitize_kwargs_for_azure(filtered_kwargs, model_name)
            use_tool_calling = "tools" in filtered_kwargs

            try:
                with _concurrency_semaphore:
                    completion = backoff_create(
                        create_completion_with_rotation,
                        OPENAI_TIMEOUT_EXCEPTIONS,
                        messages=messages,
                        **filtered_kwargs,
                    )
            except openai.BadRequestError as e2:
                # Second attempt failed - check if tool_choice is the issue
                err_msg2 = str(e2).lower()
                if _should_strip_tool_params(err_msg2):
                    logger.warning(
                        f"Tool calling rejected ({err_msg2[:150]}...). "
                        f"Falling back to plain text + JSON extraction."
                    )
                    filtered_kwargs.pop("tools", None)
                    filtered_kwargs.pop("tool_choice", None)
                    use_tool_calling = False

                    with _concurrency_semaphore:
                        completion = backoff_create(
                            create_completion_with_rotation,
                            OPENAI_TIMEOUT_EXCEPTIONS,
                            messages=messages,
                            **filtered_kwargs,
                        )
                else:
                    raise

        # If tool calling is also rejected (from initial path), strip all structured params
        elif use_tool_calling and (
            "tool_choice" in err_msg
            or "tools" in err_msg
            or "function calling" in err_msg
            or "function_call" in err_msg
            or _should_strip_tool_params(err_msg)
        ):
            logger.warning(
                f"Tool calling rejected ({err_msg[:150]}...). "
                f"Falling back to plain text + JSON extraction."
            )
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)
            use_tool_calling = False

            with _concurrency_semaphore:
                completion = backoff_create(
                    create_completion_with_rotation,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    messages=messages,
                    **filtered_kwargs,
                )
        else:
            raise

    if completion is None:
        raise RuntimeError("API call returned None. Check backoff_create logs.")

    req_time = time.time() - t0
    choice = completion.choices[0]

    # ---------------------------------------------------------
    # Parse response based on method used
    # ---------------------------------------------------------
    if func_spec is None:
        # Plain text response
        output = choice.message.content

    elif use_response_format:
        # response_format mode: content should be valid JSON
        content = choice.message.content
        if content:
            fixed_content = fix_json_string(content)
            output = safe_json_loads(fixed_content)
            if isinstance(output, dict) and "raw_output" in output:
                logger.error(
                    f"response_format JSON parse failed. Raw: {fixed_content[:500]}"
                )
            else:
                output = _validate_and_fix_output(output, func_spec)
        else:
            logger.error("response_format returned empty content.")
            output = {"raw_output": "", "error": "Empty response"}

    elif use_tool_calling:
        # Tool calling mode: extract from tool_calls
        tool_calls = getattr(choice.message, "tool_calls", None)

        if tool_calls:
            try:
                output = func_spec.parse_tool_call(tool_calls[0])
            except Exception as e:
                logger.error(
                    f"func_spec.parse_tool_call failed: {e}. "
                    f"Falling back to manual parsing."
                )
                raw_args = tool_calls[0].function.arguments
                output = safe_json_loads(fix_json_string(raw_args))
                if not isinstance(output, dict) or "raw_output" in output:
                    output = _validate_and_fix_output(output, func_spec)
        else:
            # Model ignored tool spec, returned plain text
            content = choice.message.content
            if content:
                logger.warning(
                    "Expected tool_calls but none found. "
                    "Extracting JSON from content."
                )
                raw_args = extract_json_from_content(content)
                fixed_args = fix_json_string(raw_args)
                output = safe_json_loads(fixed_args)
                if isinstance(output, dict) and "raw_output" in output:
                    logger.error(f"JSON extraction failed. Raw: {fixed_args[:500]}")
                else:
                    output = _validate_and_fix_output(output, func_spec)
            else:
                raise AssertionError(
                    f"Both tool_calls and content are empty. "
                    f"Response: {choice.message}"
                )

    else:
        # Plain text fallback: extract JSON from content
        content = choice.message.content
        if content:
            logger.warning(
                "Structured output params were stripped. "
                "Extracting JSON from plain text response."
            )
            raw_args = extract_json_from_content(content)
            fixed_args = fix_json_string(raw_args)
            output = safe_json_loads(fixed_args)
            if isinstance(output, dict) and "raw_output" not in output:
                output = _validate_and_fix_output(output, func_spec)
        else:
            raise AssertionError(
                f"Empty response after all fallbacks. Response: {choice.message}"
            )

    # Token usage
    in_tokens = getattr(completion.usage, "prompt_tokens", 0)
    out_tokens = getattr(completion.usage, "completion_tokens", 0)

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": getattr(completion, "model", "openai-unknown"),
        "created": getattr(completion, "created", None),
    }

    if func_spec is not None:
        logger.info(f"Structured output keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}")

    return output, req_time, in_tokens, out_tokens, info
