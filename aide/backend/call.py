import logging

from aide.utils.llm_caller import LLM
from aide.backend.backend_utils import PromptType
from aide.utils.config import Config

logger = logging.getLogger("aide")

def r1_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    if type(prompt) == str:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    else:
        logger.info(f"prompt: {prompt[0]['content']}\n{prompt[1]['content']}", extra={"verbose": True})
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if cfg.agent.steerable_reasoning == True:
        try:
            response = llm.stream_complete(prompt, **model_kwargs)
        except BaseException as e:
            logger.error("LLM call failed", exc_info=True)
            raise
    else:
        response = llm.stream_generate(
            prompt,
            **model_kwargs
        )


    if "</think>" in response:
        res = response[response.find("</think>")+8:]
    else:
        res = response

    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"response without think:\n{res}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res

def gpt_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model                                        
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    if type(prompt) == str:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    else:
        logger.info(f"prompt: {prompt[0]['content']}\n{prompt[1]['content']}", extra={"verbose": True})
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = llm.stream_generate(
        prompt,
        **model_kwargs
    )

    res = response
    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res