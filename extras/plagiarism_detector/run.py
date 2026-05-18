from argparse import ArgumentParser
from pathlib import Path

from analyze import run_plagiarism_detector

parser = ArgumentParser(description="Script for analyzing submissions and detecting plagiarism.")

parser.add_argument(
    "--submission",
    help="Path to the submission JSONL file",
    type=str,
    required=True,
)
parser.add_argument(
    "--output-dir",
    help="Path to the output directory",
    type=str,
    required=True,
)
parser.add_argument(
    "--timeout",
    help="Timeout for each submission processing in seconds",
    type=int,
    default=300,
    required=False,
)

args = parser.parse_args()


if __name__ == "__main__":
    run_plagiarism_detector(
        submission=Path(args.submission),
        output_dir=Path(args.output_dir),
        timeout=args.timeout,
    )
