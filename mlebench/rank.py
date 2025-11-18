import json
import re
from logging import Logger
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from mlebench.utils import get_logger, read_csv

logger = get_logger(__name__)


def _safe_path_component(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-\.]", "-", cleaned)
    return cleaned.lower()


def load_competitions(
    split_type: str,
    competition_category: str,
    splits_dir: Path,
    competition_categories_path: Path,
    logger: Logger,
) -> List[str]:
    """Load competition IDs filtered by split and competition category."""
    split_file = splits_dir / f"{split_type}.txt"

    if not split_file.exists():
        logger.error(f"Split file not found for split '{split_type}': {split_file}")
        return []

    split_competitions = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(split_competitions) == 0:
        logger.warning(f"No competitions listed in split file: {split_file}")

    competition_categories = read_csv(competition_categories_path)

    if competition_categories.empty:
        logger.error(f"Competition categories file is empty: {competition_categories_path}")
        return []

    competition_categories["category"] = competition_categories["category"].astype(str).str.strip()
    category_filter = (
        competition_categories["category"].str.casefold() == competition_category.strip().casefold()
    )
    category_competitions = set(
        competition_categories.loc[category_filter, "competition_id"].dropna().astype(str)
    )

    if len(category_competitions) == 0:
        logger.error(
            f"No competitions found with category '{competition_category}' in {competition_categories_path}"
        )
        return []

    competitions = [
        competition_id
        for competition_id in split_competitions
        if competition_id in category_competitions
    ]

    if len(competitions) == 0:
        logger.warning(
            f"No competitions from split '{split_type}' match category '{competition_category}'. "
            "Check split and category configuration."
        )

    logger.info(
        f"Loaded {len(competitions)} competitions for split '{split_type}' and category '{competition_category}'."
    )

    return competitions


def get_competition_results(
    competition_id: str, runs_dir: Path, experiment_groups: pd.DataFrame, logger: Logger
) -> pd.DataFrame | None:
    """Collect all results for a specific competition across all run groups."""
    results = []

    # Get unique run groups from experiment_groups
    run_groups = experiment_groups["run_group"].unique()

    for run_group in run_groups:
        run_group_dir = runs_dir / run_group
        if not run_group_dir.exists() or not run_group_dir.is_dir():
            continue

        # Find grading report files
        grading_reports = sorted(list(run_group_dir.glob("*grading_report*.json")))
        if len(grading_reports) == 0:
            continue

        # Load the latest grading report
        latest_report = grading_reports[-1]
        data = json.loads(latest_report.read_text())

        reports = data.get("competition_reports", [])
        if isinstance(reports, dict):
            reports = [reports]

        for report in reports:
            if report.get("competition_id") == competition_id:
                report["run_group"] = run_group
                results.append(report)
                break

    if len(results) == 0:
        logger.warning(f"No results found for competition: {competition_id}")
        return None

    results_df = pd.DataFrame(results)

    # Merge with experiment groups
    results_df = results_df.merge(experiment_groups, how="left", on="run_group")

    return results_df


def load_sample_reports(sample_report_path: Path, logger: Logger) -> dict[str, dict]:
    sample_reports: dict[str, dict] = {}
    if not sample_report_path.exists():
        logger.warning(f"Sample grading report not found at {sample_report_path}")
        return sample_reports
    try:
        sample_data = json.loads(sample_report_path.read_text())
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse sample grading report at {sample_report_path}: {exc}")
        return sample_reports

    sample_competitions = sample_data.get("competition_reports", [])
    if isinstance(sample_competitions, dict):
        sample_competitions = [sample_competitions]

    for report in sample_competitions:
        competition_id = report.get("competition_id")
        if competition_id:
            sample_reports[competition_id] = report
    return sample_reports


def score_competition_results(
    competition_id: str,
    results_df: pd.DataFrame,
    sample_reports: dict[str, dict],
    logger: Logger,
) -> pd.DataFrame:
    sample_report = sample_reports.get(competition_id)
    sample_score = sample_report.get("score") if sample_report else None
    sample_gold = sample_report.get("gold_threshold") if sample_report else None
    denominator: float | None = None
    if sample_score is not None and sample_gold is not None:
        denominator = sample_score - sample_gold
        if denominator == 0:
            logger.warning(
                "Sample submission score equals gold threshold for %s; normalized scores unavailable.",
                competition_id,
            )
            denominator = None
    else:
        logger.warning(
            "Missing sample submission baseline for %s; normalized scores unavailable.",
            competition_id,
        )

    if denominator is not None:
        results_df["normalized_score"] = (sample_score - results_df["score"]) / denominator
    else:
        results_df["normalized_score"] = pd.NA

    stats_df = results_df.groupby("experiment_id", group_keys=True).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        n_runs=("score", "count"),
        mean_normalized_score=("normalized_score", "mean"),
        std_normalized_score=("normalized_score", "std"),
        mean_any_medal=("any_medal", "mean"),
        std_any_medal=("any_medal", "std"),
    )
    stats_df = stats_df.reset_index()

    stats_df = stats_df.sort_values(
        "mean_normalized_score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    return stats_df


def aggregate_scores(rank_series_list: list[pd.Series], name: str) -> pd.DataFrame | None:
    if len(rank_series_list) == 0:
        return None
    rank_df = pd.concat(rank_series_list, axis=1).fillna(0)
    rank_df = rank_df.agg(["mean", "std"], axis=1)
    rank_df.columns = [f"mean_{name}", f"std_{name}"]
    return rank_df


def collect_rankings(
    run_group_experiments_path: Path,
    runs_dir: Path,
    splits_dir: Path,
    competition_categories_path: Path,
    split_type: str,
    competition_category: str,
    experiment_agents_path: Path,
    output_dir: Path,
    sample_report_path: Path,
):
    sample_reports = load_sample_reports(sample_report_path, logger)

    # Load competitions from configured split and category
    competitions = load_competitions(
        split_type=split_type,
        competition_category=competition_category,
        splits_dir=splits_dir,
        competition_categories_path=competition_categories_path,
        logger=logger,
    )

    if len(competitions) == 0:
        logger.error("No competitions available to process. Exiting.")
        return
    logger.info(f"Competitions to evaluate: {competitions}")

    output_dir.mkdir(parents=True, exist_ok=True)

    split_dirname = _safe_path_component(split_type)
    category_dirname = _safe_path_component(competition_category)
    competition_results_dir = output_dir / split_dirname / category_dirname / "competition_results"
    competition_results_dir.mkdir(parents=True, exist_ok=True)
    base_output_dir = competition_results_dir.parent

    logger.info(f"Writing results under {base_output_dir}")

    # Read experiment to run_group mapping
    experiment_groups = read_csv(run_group_experiments_path)
    logger.info(f"Found {len(experiment_groups)} experiment-run_group mappings")

    # Read experiment agents mapping
    experiment_agents = read_csv(experiment_agents_path)
    logger.info(f"Found {len(experiment_agents)} experiment-agent mappings")

    # Collect results for each tabular competition
    rank_series_list: list[pd.Series] = []
    medal_series_list: list[pd.Series] = []

    for competition_id in competitions:
        logger.info(f"Processing competition: {competition_id}")
        results_df = get_competition_results(competition_id, runs_dir, experiment_groups, logger)

        if results_df is None or len(results_df) == 0:
            logger.warning(f"No results for {competition_id}")
            continue

        # Filter to only valid scores
        results_df = results_df[results_df["score"].notna()].copy()

        if len(results_df) == 0:
            logger.warning(f"No valid scores for {competition_id}")
            continue

        stats_df = score_competition_results(competition_id, results_df, sample_reports, logger)

        # Save per-competition CSV (without agent descriptions)
        competition_output = competition_results_dir / f"{competition_id}.csv"
        stats_df.to_csv(competition_output, index=False)
        logger.info(f"Saved results for {competition_id} to {competition_output}")

        # Collect ranks for overall ranking
        rank_series_list.append(
            stats_df.set_index("experiment_id")["mean_normalized_score"].rename(competition_id)
        )
        medal_series_list.append(
            stats_df.set_index("experiment_id")["mean_any_medal"].rename(competition_id)
        )

    # Create overall score DataFrame
    rank_df = aggregate_scores(rank_series_list, "normalized_score")
    medal_df = aggregate_scores(medal_series_list, "any_medal")
    if rank_df is None:
        logger.error("No scores collected for any competition!")
        return

    # Create final results DataFrame
    final_results = pd.concat([rank_df, medal_df], axis=1).reset_index()
    final_results = final_results.merge(experiment_agents, on="experiment_id", how="left")

    # Sort by mean_rank
    final_results = final_results.sort_values("mean_normalized_score", ascending=False).reset_index(
        drop=True
    )

    # Save final mean ranks CSV
    final_output = base_output_dir / "overall_ranks.csv"
    final_results.to_csv(final_output, index=False)
    logger.info(f"Saved final ranking to {final_output}")
