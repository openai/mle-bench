import asyncio
from argparse import ArgumentParser
from pathlib import Path

from analyze import run_analysis
from prompts import ALL_QUESTIONS

parser = ArgumentParser(description="Script for analyzing submissions and detecting plagiarism.")

parser.add_argument(
    "--submission",
    help="Path to the JSONL file of submissions. Format mimics default submission format, with `logs_path` and/or `code_path` instead of `submission_path`.",
    type=str,
    required=True,
)
parser.add_argument(
    "--output-dir",
    help="Path to the directory where the analysis CSV will be saved.",
    type=str,
    required=True,
)
parser.add_argument(
    "--questions",
    help="List of question IDs to ask for each log file defined in `prompts.py`. Defaults to all questions.",
    type=str,
    nargs="+",
    default=list(ALL_QUESTIONS.keys()),
    required=False,
)
parser.add_argument(
    "--extra-prompt",
    help="Path to plaintext file containing additional prompt to be concatenated to end of the base analysis prompt. Useful for e.g. different log formats across agents.",
    default=None,
    required=False,
)

args = parser.parse_args()


if __name__ == "__main__":
    asyncio.run(
        run_analysis(
            submission=Path(args.submission),
            questions=args.questions,
            output_dir=Path(args.output_dir),
            extra_prompt=(Path(args.extra_prompt) if args.extra_prompt else None),
        )
    )
