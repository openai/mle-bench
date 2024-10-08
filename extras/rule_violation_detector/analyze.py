import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import openai
import pandas as pd
import tenacity
from openai import AsyncOpenAI as OpenAI
from prompts import AFFIRMATIVE_FLAG, ALL_QUESTIONS, NEGATIVE_FLAG, PREAMBLE_CODE, PREAMBLE_LOGS
from tqdm import tqdm

from mlebench.utils import get_logger, get_timestamp, purple, read_jsonl

logger = get_logger(__name__)

# Create client on request to avoid requiring API key when this module is just imported
_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )
        )
        # dont want info logs from httpx (this is called by openai on every request)
        httpx_logger: logging.Logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.WARNING)
    return _openai_client


async def run_analysis(
    submission: Path,
    questions: list[str],
    output_dir: Path,
    extra_prompt: Optional[Path] = None,
) -> None:
    """Analyzes a run of MLE-bench and saves the results to a CSV file"""
    submissions = read_jsonl(submission.as_posix())

    if extra_prompt is not None:
        with open(extra_prompt, "r") as file:
            extra_prompt = file.read()

    analysis_df = await analyze_submissions(submissions, questions, extra_prompt)

    timestamp = get_timestamp()
    save_path = output_dir / f"analysis_{timestamp}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(save_path, index=False, na_rep="")  # None values saved as empty strings
    logger.info(purple(f"Saved analysis to {save_path}"))

    # Add summary of binarized questions; log and save
    binary_summary = get_binary_column_means(analysis_df)
    summary_dict = {
        question.replace("_binarized", ""): f"{percentage:.1f}%"
        for question, percentage in binary_summary.items()
    }
    summary_json = json.dumps(summary_dict, indent=2)
    logger.info(f"Summary of binarized questions (percentage True):\n{summary_json}")
    summary_save_path = output_dir / f"analysis_summary_{timestamp}.json"
    with open(summary_save_path, "w") as file:
        file.write(summary_json)


async def analyze_submissions(
    submissions: list[dict],
    question_ids: list[str],
    extra_prompt: Optional[str] = None,
    max_chunk_size: int = 128_000,
):
    """
    Analyzes logs and code from each submission across a set of questions.
    """
    analysis_rows = []

    logger.info(f"Analyzing {len(submissions)} submissions")

    tasks = [
        analyze_single_submission(submission, question_ids, extra_prompt, max_chunk_size)
        for submission in submissions
    ]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Analyzing submissions"):
        row = await f
        analysis_rows.append(row)

    analysis_df = pd.DataFrame(analysis_rows)
    return analysis_df


async def analyze_single_submission(
    submission: dict,
    question_ids: list[str],
    extra_prompt: Optional[str],
    max_chunk_size: int,
) -> dict:

    assert "competition_id" in submission and isinstance(submission["competition_id"], str)

    competition_id = submission["competition_id"]

    logger.info(f"--- Analyzing submission for {competition_id} ---")

    row = {
        "competition_id": competition_id,
        "submission_path": submission["submission_path"],
    }

    logs_content = ""
    code_content = ""

    def _read_content(path: Path) -> str:
        """Read logs or code content from a file or directory"""
        content = ""
        if path.is_dir():
            logger.warning(f"Path is a directory: {path}. Concatenating all relevant files.")
            for file in path.glob("**/*"):
                if file.suffix in [".log", ".py"]:
                    with open(file, "r") as f:
                        content += f"--- {file.name} ---\n"
                        content += f.read() + "\n"
        else:
            with open(path, "r") as file:
                content = file.read()
        return content

    # Handle logs
    if "logs_path" in submission and submission["logs_path"]:
        logs_path = Path(submission["logs_path"])
        if logs_path.exists():
            logs_content = _read_content(logs_path)
            logger.info(f"Logs word count: {len(logs_content.split())}")
        else:
            logger.warning(f"Logs path not found: {logs_path}")
    else:
        logger.warning("Logs path not provided in submission")

    # Handle code
    if "code_path" in submission and submission["code_path"]:
        code_path = Path(submission["code_path"])
        if code_path.exists():
            code_content = _read_content(code_path)
            logger.info(f"Code word count: {len(code_content.split())}")
        else:
            logger.warning(f"Code path not found: {code_path}")
    else:
        logger.warning("Code path not provided in submission")

    tasks = []
    for question_id in question_ids:
        question_meta = ALL_QUESTIONS[question_id]
        content_type = question_meta["type"]

        if content_type == "logs" and not logs_content:
            logger.warning(f"Skipping question {question_id} due to missing logs")
            continue
        elif content_type == "code" and not code_content:
            logger.warning(f"Skipping question {question_id} due to missing code")
            continue

        content = logs_content if content_type == "logs" else code_content
        question = question_meta["question"]

        logger.info(f"Analyzing {content_type} for question: {question_id}")
        tasks.append(
            process_question(
                content,
                content_type,
                question,
                question_id,
                extra_prompt,
                max_chunk_size,
            )
        )

    results = await asyncio.gather(*tasks)
    for result in results:
        row.update(result)

    return row


async def process_question(
    content: str,
    content_type: str,
    question: str,
    question_id: str,
    extra_prompt: Optional[str],
    max_chunk_size: int,
) -> dict:
    try:
        if len(content) <= max_chunk_size:
            answer = (
                await get_answer_from_analyzer(content, content_type, question, extra_prompt)
            ).strip()
            logger.debug(f"Answer: {answer}")
            result = {
                f"{content_type}_{question_id}": answer,
                f"{content_type}_{question_id}_binarized": None,
            }
            if answer.strip().endswith(AFFIRMATIVE_FLAG):
                result[f"{content_type}_{question_id}_binarized"] = True
            elif answer.strip().endswith(NEGATIVE_FLAG):
                result[f"{content_type}_{question_id}_binarized"] = False
        else:
            logger.warning(f"Text exceeds model token limit. Applying question in chunks.")
            # Process chunks in parallel
            tasks = []
            binarized = []
            for chunk_start in range(0, len(content), max_chunk_size):
                content_chunk = content[chunk_start : chunk_start + max_chunk_size]
                tasks.append(
                    get_answer_from_analyzer(content_chunk, content_type, question, extra_prompt)
                )

            answers = await asyncio.gather(*tasks)
            answers = [answer.strip() for answer in answers]
            for answer in answers:
                logger.debug(f"Answer: {answer}")
                if answer.strip().endswith(AFFIRMATIVE_FLAG):
                    binarized.append(True)
                elif answer.strip().endswith(NEGATIVE_FLAG):
                    binarized.append(False)
                else:
                    binarized.append(None)

            result = {
                f"{content_type}_{question_id}": "CHUNKED_ANSWERS:\n\n"
                + "\n\n---------\n\n".join(answers),
                f"{content_type}_{question_id}_binarized": (
                    None if all([el is None for el in binarized]) else any(binarized)
                ),
            }
        return result
    except openai.BadRequestError as e:
        error_msg = f"Error analyzing {content_type} for question {question_id}: {e}"
        logger.error(error_msg)
        return {f"{content_type}_{question_id}": error_msg}


@tenacity.retry(
    wait=tenacity.wait_exponential(
        multiplier=1, min=2, max=60
    ),  # Exponential backoff starting at 2 seconds, max 60 seconds
    stop=tenacity.stop_after_attempt(100),  # Stop after 100 attempts
    retry=tenacity.retry_if_exception_type(
        (
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
        )
    ),
    before=tenacity.before_log(logger, logging.DEBUG),
)
async def openai_chat_completion_with_retries(*args, **kwargs):
    client = get_openai_client()
    return await client.chat.completions.create(*args, **kwargs)


async def get_answer_from_analyzer(
    content: str, content_type: str, question: str, extra_prompt: Optional[str] = None
) -> str:
    """Ask a question to the analyzer LLM"""
    PREAMBLE = PREAMBLE_LOGS if content_type == "logs" else PREAMBLE_CODE

    extra_prompt = extra_prompt or ""

    completion = await openai_chat_completion_with_retries(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PREAMBLE + extra_prompt},
            {"role": "user", "content": content},
            {"role": "system", "content": question},
        ],
        temperature=0.0,
    )

    return completion.choices[0].message.content


def get_binary_column_means(dataframe: pd.DataFrame):
    """
    Get mean of all binary columns, counting True as 1 and False/None as 0
    """
    binary_cols = [col for col in dataframe.columns if col.endswith("_binarized")]
    report = dataframe[binary_cols].fillna(False).mean() * 100  # convert to percentage
    return report
