"""High-level grading functionality"""
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from mlebench.data import get_leaderboard, is_dataset_prepared
from mlebench.grade_helpers import CompetitionReport
from mlebench.registry import Competition, Registry
from mlebench.registry import registry as DEFAULT_REGISTRY
from mlebench.utils import get_logger, get_timestamp, load_answers, purple, read_csv, read_jsonl

logger = get_logger(__name__)


def grade_jsonl(
    path_to_submissions: Path,
    output_dir: Path,
    registry: Registry = DEFAULT_REGISTRY,
):
    """
    Grades multiple submissions stored in a JSONL file.
    Saves the aggregated report as a JSON file.
    """

    submissions = read_jsonl(path_to_submissions, skip_commented_out_lines=True)
    competitions_reports = []

    for submission in tqdm(submissions, desc="Grading submissions", unit="submission"):
        submission_path = Path(str(submission["submission_path"]))
        competition_id = submission["competition_id"]
        competition = registry.get_competition(competition_id)
        single_report = grade_csv(submission_path, competition)
        competitions_reports.append(single_report)

    aggregated_report = aggregate_reports(competitions_reports)
    timestamp = get_timestamp()
    save_path = output_dir / f"{timestamp}_grading_report.json"
    logger.info(
        json.dumps(
            {k: v for k, v in aggregated_report.items() if k != "competition_reports"}, indent=4
        )
    )

    output_dir.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(aggregated_report, f, indent=2)
    logger.info(purple(f"Saved summary report to {save_path}"))


def grade_csv(path_to_submission: Path, competition: Competition) -> CompetitionReport:
    """Grades a submission CSV for the given competition."""

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
        )

    score = None
    submission_exists = path_to_submission.is_file() and path_to_submission.suffix.lower() == ".csv"

    if submission_exists:
        submission_df = read_csv(path_to_submission)
        answers = load_answers(competition.answers)
        score = competition.grader(submission_df, answers)
    else:
        logger.warning(
            f"Invalid submission file: {path_to_submission}. Please check that the file exists and it is a CSV."
        )

    valid_submission = score is not None
    competition_leaderboard = get_leaderboard(competition)
    rank_info = competition.grader.rank_score(score, competition_leaderboard)
    is_lower_better = competition.grader.is_lower_better(competition_leaderboard)

    return CompetitionReport(
        competition_id=competition.id,
        score=score,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"],
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        is_lower_better=is_lower_better,
        created_at=datetime.now(),
        submission_path=str(path_to_submission),
    )


def validate_submission(submission: Path, competition: Competition) -> tuple[bool, str]:
    """
    Validates a submission for the given competition by actually running the competition grader.
    This is designed for end users, not developers (we assume that the competition grader is
    correctly implemented and use that for validating the submission, not the other way around).
    """
    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."

    if not submission.suffix.lower() == ".csv":
        return False, "Submission invalid! Submission file must be a CSV file."

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
        )

    try:
        competition.grader.grade_fn(read_csv(submission), read_csv(competition.answers))
    except Exception as e:
        return (
            False,
            f"Submission invalid! The attempt to grade the submission has resulted in the following error message:\n{e}",
        )

    return True, "Submission is valid."


def aggregate_reports(competition_reports: list[CompetitionReport]) -> dict:
    """
    Builds the summary report from a list of competition reports.
    If pass_at_n is True, then aggregate performence of competitions by selecting the best performance per
    competition, otherwise sum metrics
    """

    total_gold_medals = sum(report.gold_medal for report in competition_reports)
    total_silver_medals = sum(report.silver_medal for report in competition_reports)
    total_bronze_medals = sum(report.bronze_medal for report in competition_reports)
    total_above_median = sum(report.above_median for report in competition_reports)
    total_submissions = sum(report.submission_exists for report in competition_reports)
    total_valid_submissions = sum(report.valid_submission for report in competition_reports)

    summary_report = {
        "total_runs": int(len(competition_reports)),
        "total_runs_with_submissions": int(total_submissions),
        "total_valid_submissions": int(total_valid_submissions),
        "total_medals": int(total_gold_medals + total_silver_medals + total_bronze_medals),
        "total_gold_medals": int(total_gold_medals),
        "total_silver_medals": int(total_silver_medals),
        "total_bronze_medals": int(total_bronze_medals),
        "total_above_median": int(total_above_median),
        "competition_reports": [cr.to_dict() for cr in competition_reports],
    }

    return summary_report
