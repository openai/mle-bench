import argparse
import json
from pathlib import Path

from mlebench.data import download_and_prepare_dataset, ensure_leaderboard_exists
from mlebench.grade import grade_csv, grade_jsonl
from mlebench.registry import registry
from mlebench.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Runs agents on Kaggle competitions.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run.")

    # Prepare sub-parser
    parser_prepare = subparsers.add_parser(
        name="prepare",
        help="Download and prepare competitions for the MLE-bench dataset.",
    )
    parser_prepare.add_argument(
        "-c",
        "--competition-id",
        help=f"ID of the competition to prepare. Valid options: {registry.list_competition_ids()}",
        type=str,
        required=False,
    )
    parser_prepare.add_argument(
        "-a",
        "--all",
        help="Prepare all competitions.",
        action="store_true",
    )
    parser_prepare.add_argument(
        "--lite",
        help="Prepare all the low complexity competitions (MLE-bench Lite).",
        action="store_true",
        required=False,
    )
    parser_prepare.add_argument(
        "-l",
        "--list",
        help="Prepare a list of competitions specified line by line in a text file.",
        type=str,
        required=False,
    )
    parser_prepare.add_argument(
        "--keep-raw",
        help="Keep the raw competition files after the competition has been prepared.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )
    parser_prepare.add_argument(
        "--overwrite-checksums",
        help="[For Developers] Overwrite the checksums file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--overwrite-leaderboard",
        help="[For Developers] Overwrite the leaderboard file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--skip-verification",
        help="[For Developers] Skip the verification of the checksums.",
        action="store_true",
        required=False,
        default=False,
    )

    # Grade eval sub-parser
    parser_grade_eval = subparsers.add_parser(
        "grade",
        help="Grade a submission to the eval, comprising of several competition submissions",
    )
    parser_grade_eval.add_argument(
        "--submission",
        help="Path to the JSONL file of submissions. Refer to README.md#submission-format for the required format.",
        type=str,
        required=True,
    )
    parser_grade_eval.add_argument(
        "--output-dir",
        help="Path to the directory where the evaluation metrics will be saved.",
        type=str,
        required=True,
    )
    parser_grade_eval.add_argument(
        "--data-dir",
        help="Path to the directory where the data used for grading is stored.",
        required=False,
        default=registry.get_data_dir(),
    )

    # Grade sample sub-parser
    parser_grade_sample = subparsers.add_parser(
        name="grade-sample",
        help="Grade a single sample (competition) in the eval",
    )
    parser_grade_sample.add_argument(
        "submission",
        help="Path to the submission CSV file.",
        type=str,
    )
    parser_grade_sample.add_argument(
        "competition_id",
        help=f"ID of the competition to grade. Valid options: {registry.list_competition_ids()}",
        type=str,
    )
    parser_grade_sample.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )

    # Dev tools sub-parser
    parser_dev = subparsers.add_parser("dev", help="Developer tools for extending MLE-bench.")
    dev_subparsers = parser_dev.add_subparsers(dest="dev_command", help="Developer command to run.")

    # Set up 'download-leaderboard' under 'dev'
    parser_download_leaderboard = dev_subparsers.add_parser(
        "download-leaderboard",
        help="Download the leaderboard for a competition.",
    )
    parser_download_leaderboard.add_argument(
        "-c",
        "--competition-id",
        help=f"Name of the competition to download the leaderboard for. Valid options: {registry.list_competition_ids()}",
        type=str,
        required=False,
    )
    parser_download_leaderboard.add_argument(
        "--all",
        help="Download the leaderboard for all competitions.",
        action="store_true",
    )
    parser_download_leaderboard.add_argument(
        "--force",
        help="Force download the leaderboard, even if it already exists.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        new_registry = registry.set_data_dir(Path(args.data_dir))

        if args.lite:
            competitions = [
                new_registry.get_competition(competition_id)
                for competition_id in new_registry.get_lite_competition_ids()
            ]
        elif args.all:
            competitions = [
                new_registry.get_competition(competition_id)
                for competition_id in registry.list_competition_ids()
            ]
        elif args.list:
            with open(args.list, "r") as f:
                competition_ids = f.read().splitlines()
            competitions = [
                new_registry.get_competition(competition_id) for competition_id in competition_ids
            ]
        else:
            if not args.competition_id:
                parser_prepare.error(
                    "One of --lite, --all, --list, or --competition-id must be specified."
                )
            competitions = [new_registry.get_competition(args.competition_id)]

        for competition in competitions:
            download_and_prepare_dataset(
                competition=competition,
                keep_raw=args.keep_raw,
                overwrite_checksums=args.overwrite_checksums,
                overwrite_leaderboard=args.overwrite_leaderboard,
                skip_verification=args.skip_verification,
            )
    if args.command == "grade":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        submission = Path(args.submission)
        output_dir = Path(args.output_dir)
        grade_jsonl(submission, output_dir, new_registry)
    if args.command == "grade-sample":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        competition = new_registry.get_competition(args.competition_id)
        submission = Path(args.submission)
        report = grade_csv(submission, competition)
        logger.info("Competition report:")
        logger.info(json.dumps(report.to_dict(), indent=4))
    if args.command == "dev":
        if args.dev_command == "download-leaderboard":
            if args.all:
                for competition_id in registry.list_competition_ids():
                    competition = registry.get_competition(competition_id)
                    ensure_leaderboard_exists(competition, force=args.force)
            elif args.competition_id:
                competition = registry.get_competition(args.competition_id)
                ensure_leaderboard_exists(competition, force=args.force)
            else:
                parser_download_leaderboard.error(
                    "Either --all or --competition-id must be specified."
                )


if __name__ == "__main__":
    main()
