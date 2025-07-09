import argparse
import json
from pathlib import Path

from researchbench.data import download_and_prepare_dataset, ensure_leaderboard_exists
from researchbench.grade import grade_csv, grade_jsonl
from researchbench.registry import registry
from researchbench.utils import get_logger
from researchbench.research_tasks.manager import prepare_research_task, list_research_tasks, run_research_task

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ResearchBench: A platform for evaluating AI systems on research tasks.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run.")

    # Prepare sub-parser for ML competitions
    parser_prepare = subparsers.add_parser(
        name="prepare",
        help="Download and prepare competitions for the ML dataset.",
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

    # Research tasks sub-parser
    parser_research = subparsers.add_parser(
        name="research",
        help="Manage research tasks",
    )
    research_subparsers = parser_research.add_subparsers(dest="research_command", help="Research command to run.")

    # List research tasks
    parser_list_research = research_subparsers.add_parser(
        "list",
        help="List available research tasks",
    )
    parser_list_research.add_argument(
        "--category",
        help="Filter tasks by category",
        type=str,
        required=False,
    )

    # Prepare research task
    parser_prepare_research = research_subparsers.add_parser(
        "prepare",
        help="Prepare a research task",
    )
    parser_prepare_research.add_argument(
        "task_id",
        help="ID of the research task to prepare",
        type=str,
    )
    parser_prepare_research.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored",
        type=str,
        required=False,
        default=Path.home() / ".researchbench" / "data",
    )

    # Run research task
    parser_run_research = research_subparsers.add_parser(
        "run",
        help="Run a research task",
    )
    parser_run_research.add_argument(
        "task_id",
        help="ID of the research task to run",
        type=str,
    )
    parser_run_research.add_argument(
        "--output-dir",
        help="Path to the directory where the output will be stored",
        type=str,
        required=False,
        default=Path.home() / ".researchbench" / "output",
    )
    parser_run_research.add_argument(
        "--data-dir",
        help="Path to the directory where the data is stored",
        type=str,
        required=False,
        default=Path.home() / ".researchbench" / "data",
    )
    parser_run_research.add_argument(
        "--params",
        help="JSON string with parameters for the research task",
        type=str,
        required=False,
        default="{}",
    )

    # Dev tools sub-parser
    parser_dev = subparsers.add_parser("dev", help="Developer tools for extending ResearchBench.")
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
    elif args.command == "grade":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        submission = Path(args.submission)
        output_dir = Path(args.output_dir)
        grade_jsonl(submission, output_dir, new_registry)
    elif args.command == "grade-sample":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        competition = new_registry.get_competition(args.competition_id)
        submission = Path(args.submission)
        report = grade_csv(submission, competition)
        logger.info("Competition report:")
        logger.info(json.dumps(report.to_dict(), indent=4))
    elif args.command == "research":
        if args.research_command == "list":
            category = args.category if hasattr(args, "category") else None
            tasks = list_research_tasks(category)
            logger.info("Available research tasks:")
            for task in tasks:
                logger.info(f"- {task['id']}: {task['name']} ({task['category']})")
        elif args.research_command == "prepare":
            data_dir = Path(args.data_dir)
            prepare_research_task(args.task_id, data_dir)
            logger.info(f"Research task {args.task_id} prepared successfully.")
        elif args.research_command == "run":
            data_dir = Path(args.data_dir)
            output_dir = Path(args.output_dir)
            params = json.loads(args.params)
            result = run_research_task(args.task_id, data_dir, output_dir, params)
            logger.info(f"Research task {args.task_id} completed successfully.")
            logger.info(f"Results: {json.dumps(result, indent=4)}")
        else:
            parser_research.error("One of list, prepare, or run must be specified.")
    elif args.command == "dev":
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