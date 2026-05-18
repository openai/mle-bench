"""
Illustrative script for how one may collate the logs. Certain agents may have multiple
log files per run. You could use something similar posthoc_concat.sh to concatenate the
logs into a single file per run. This script is not used in the benchmarking process.
"""

import argparse
import json
from pathlib import Path


def get_most_recent_non_empty_folder(base_path: Path):
    timestamped_folders = sorted(base_path.iterdir(), key=lambda p: p.name, reverse=True)
    for folder in timestamped_folders:
        logs_folder = folder / "logs"
        if logs_folder.is_dir() and any(logs_folder.iterdir()):
            return logs_folder
    return None


def main(runs_dir: Path, log_file_name: str, output: Path):
    with open(output, "w") as f:
        for comp_folder in runs_dir.iterdir():
            if comp_folder.is_dir():
                competition_id = comp_folder.name.split("_")[2]
                recent_logs_folder = get_most_recent_non_empty_folder(comp_folder)
                if recent_logs_folder:
                    log_path = recent_logs_folder / log_file_name
                    line = {
                        "competition_id": competition_id,
                        "logs_path": log_path.absolute().as_posix(),
                    }
                    f.write(f"{json.dumps(line)}\n")
                else:
                    print(f"No logs found for {comp_folder.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes a logs.jsonl for mlebench analyze from a mlebench run `runs/` directory"
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        help="Path to the `runs/` directory",
        default="runs/",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output jsonl file",
        default="logs.jsonl",
    )
    parser.add_argument(
        "--log-file-name",
        type=str,
        default="full_log.txt",
    )

    args = parser.parse_args()

    main(Path(args.runs_dir), args.log_file_name, Path(args.output))
