from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_GROUP_CSV = PROJECT_ROOT / "runs" / "run_group_experiments.csv"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_SPLIT = "split75"


def zero_report(competition_id: str) -> CompetitionReport:
    """Return a placeholder report representing no submission/medal."""
    return CompetitionReport(
        competition_id=competition_id,
        gold_medal=False,
        silver_medal=False,
        bronze_medal=False,
        above_median=False,
        submission_exists=False,
        valid_submission=False,
    )


@dataclass(frozen=True)
class CompetitionReport:
    competition_id: str
    gold_medal: bool
    silver_medal: bool
    bronze_medal: bool
    above_median: bool
    submission_exists: bool
    valid_submission: bool


@dataclass(frozen=True)
class SeedMetrics:
    total_competitions: int
    submission_exists_percentage: float
    valid_submissions_percentage: float
    above_median_percentage: float
    gold_medal_percentage: float
    silver_medal_percentage: float
    bronze_medal_percentage: float
    any_medal_percentage: float


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    ci_lower: float
    ci_upper: float
    standard_error: float


@dataclass(frozen=True)
class AggregateMetrics:
    num_seeds_averaged: int
    num_competitions_per_seed: int
    metrics: dict[str, AggregateMetric]


def build_complete_seeds(
    competition_reports: list[CompetitionReport],
    competition_ids: list[str],
    requested_seeds: int,
    pad_missing: bool = False,
) -> list[list[CompetitionReport]]:
    """
    Construct seeds such that each seed contains exactly one report per competition_id.

    Default (pad_missing=False) keeps only fully complete seeds using the minimum
    available report count across competitions. With pad_missing=True, missing
    competitions in a seed are filled with zeroed placeholder reports; the number
    of seeds is then based on the maximum available reports (or the requested limit).
    """

    comp_to_reports: dict[str, list[CompetitionReport]] = defaultdict(list)
    for report in competition_reports:
        comp_to_reports[report.competition_id].append(report)

    # Ensure every expected competition is present
    missing = [cid for cid in competition_ids if cid not in comp_to_reports]
    if missing and not pad_missing:
        raise ValueError(
            f"Missing reports for competitions: {', '.join(missing)}; cannot build complete seeds."
        )
    for cid in missing:
        comp_to_reports[cid] = []

    # Keep the reports in the order they were read (run-group order). Do not
    # sort by validity because that would permute runs independently per
    # competition and break seed alignment when some runs are invalid for only
    # a subset of competitions.

    # count available reports per competition
    comp_to_report_count = {cid: len(comp_to_reports[cid]) for cid in competition_ids}
    if not comp_to_report_count:
        raise ValueError("No competitions provided.")

    if pad_missing:
        max_available = max(comp_to_report_count.values())
        if max_available == 0:
            raise ValueError("No reports available to form any seed.")
        num_seeds = max_available if requested_seeds == -1 else min(requested_seeds, max_available)
    else:
        min_available = min(comp_to_report_count.values())
        if min_available == 0:
            raise ValueError("No reports available to form any seed.")
        num_seeds = min_available if requested_seeds == -1 else min(requested_seeds, min_available)

    if num_seeds <= 0:
        raise ValueError("No seeds requested or available after evaluating competition coverage.")

    seeds: list[list[CompetitionReport]] = []
    for idx in range(num_seeds):
        seed_reports: list[CompetitionReport] = []
        for cid in competition_ids:
            reports_for_comp = comp_to_reports[cid]
            if idx < len(reports_for_comp):
                seed_reports.append(reports_for_comp[idx])
            elif pad_missing:
                seed_reports.append(zero_report(cid))
            else:
                # Should be unreachable because num_seeds is bounded by min_available
                raise ValueError(f"Missing report for competition {cid} in seed {idx + 1}.")
        seeds.append(seed_reports)

    return seeds


def calculate_metrics_for_a_seed(competitions: list[CompetitionReport]) -> SeedMetrics:
    total_competitions = len(competitions)
    any_medals = sum(
        1 for comp in competitions if comp.gold_medal or comp.silver_medal or comp.bronze_medal
    )
    gold_medals = sum(1 for comp in competitions if comp.gold_medal)
    silver_medals = sum(1 for comp in competitions if comp.silver_medal)
    bronze_medals = sum(1 for comp in competitions if comp.bronze_medal)
    above_median = sum(1 for comp in competitions if comp.above_median)
    submission_exists = sum(1 for comp in competitions if comp.submission_exists)
    valid_submissions = sum(1 for comp in competitions if comp.valid_submission)

    return SeedMetrics(
        total_competitions=total_competitions,
        submission_exists_percentage=submission_exists / total_competitions * 100,
        valid_submissions_percentage=valid_submissions / total_competitions * 100,
        above_median_percentage=above_median / total_competitions * 100,
        gold_medal_percentage=gold_medals / total_competitions * 100,
        silver_medal_percentage=silver_medals / total_competitions * 100,
        bronze_medal_percentage=bronze_medals / total_competitions * 100,
        any_medal_percentage=any_medals / total_competitions * 100,
    )


def average_metrics_across_seeds(
    seed_metrics: list[SeedMetrics], expected_comps: int
) -> AggregateMetrics:
    if not seed_metrics:
        raise ValueError(
            "No complete seeds found; each seed must have the expected number of competitions."
        )

    all_values: defaultdict[str, list[float]] = defaultdict(list)

    for metric in seed_metrics:
        metric_dict = asdict(metric)
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                all_values[key].append(float(value))

    metrics_summary: dict[str, AggregateMetric] = {}
    for key, values in all_values.items():
        mean = float(np.mean(values))
        sem = float(stats.sem(values, ddof=1)) if len(values) > 1 else 0.0
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        metrics_summary[key] = AggregateMetric(
            mean=mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            standard_error=sem,
        )

    return AggregateMetrics(
        num_seeds_averaged=len(seed_metrics),
        num_competitions_per_seed=expected_comps,
        metrics=metrics_summary,
    )


def get_competition_reports(grading_reports: list[str]) -> list[CompetitionReport]:
    all_reports: list[CompetitionReport] = []
    for report_path in sorted(grading_reports):
        with open(report_path, "r") as f:
            grading_report = json.load(f)
        for report in sorted(
            grading_report["competition_reports"], key=lambda r: str(r["competition_id"])
        ):
            all_reports.append(
                CompetitionReport(
                    competition_id=str(report["competition_id"]),
                    gold_medal=bool(report["gold_medal"]),
                    silver_medal=bool(report["silver_medal"]),
                    bronze_medal=bool(report["bronze_medal"]),
                    above_median=bool(report["above_median"]),
                    submission_exists=bool(report["submission_exists"]),
                    valid_submission=bool(report["valid_submission"]),
                )
            )
    return all_reports


def load_run_groups_for_experiment(experiment_id: str) -> list[str]:
    """Return the run_group names for a given experiment id."""
    run_groups: list[str] = []
    with RUN_GROUP_CSV.open() as f:
        for line in f:
            if not line.strip() or line.startswith("experiment_id"):
                continue
            exp, run_group = line.strip().split(",", maxsplit=1)
            exp = exp.strip()
            run_group = run_group.strip()
            if exp == experiment_id:
                run_groups.append(run_group)
    if not run_groups:
        raise ValueError(f"Experiment id '{experiment_id}' not found in {RUN_GROUP_CSV}")
    return run_groups


def resolve_grading_report_path(run_group: str) -> Path | None:
    """Pick the single top-level grading report for a run group; warn/skip if missing."""
    run_dir = RUNS_DIR / run_group
    if not run_dir.is_dir():
        print(f"Run group directory missing, skipping: {run_group}")
        return None

    reports = sorted(p for p in run_dir.glob("*grading_report*.json") if p.is_file())

    if len(reports) == 0:
        print(f"No grading report found for run group {run_group}; skipping.")
        return None

    if len(reports) > 1:
        raise ValueError(
            f"Expected one grading report in {run_group}, found {len(reports)}: "
            f"{', '.join(str(r.name) for r in reports)}"
        )

    return reports[0]


def grading_reports_from_experiment(experiment_id: str) -> list[str]:
    """Collect grading report paths for all run groups of an experiment."""
    run_groups = sorted(load_run_groups_for_experiment(experiment_id))
    reports: list[str] = []
    for run_group in run_groups:
        report_path = resolve_grading_report_path(run_group)
        if report_path:
            reports.append(str(report_path))
    if not reports:
        raise ValueError(f"No grading reports found for experiment id '{experiment_id}'.")
    return reports


def resolve_split_path(split: str) -> Path:
    """Resolve a split name or path to an existing file under experiments/splits/."""
    split_path = Path(split)
    if not split_path.suffix:
        split_path = Path("experiments") / "splits" / f"{split}.txt"
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return split_path


def load_competition_ids(split: str) -> list[str]:
    """Load the ordered list of competitions for the given split."""
    split_path = resolve_split_path(split)
    with split_path.open() as f:
        return [line.strip() for line in f if line.strip()]


def select_seeds(
    seeds: list[list[CompetitionReport]], num_seeds: int
) -> list[list[CompetitionReport]]:
    """Return the requested number of seeds, or all if num_seeds == -1."""
    if num_seeds == -1:
        return seeds
    if num_seeds > len(seeds):
        raise ValueError(
            f"Requested {num_seeds} seeds, but only {len(seeds)} complete seeds available."
        )
    return seeds[:num_seeds]


def main(grading_reports: list[str], num_seeds: int, split: str, pad_missing: bool) -> None:
    competition_reports = get_competition_reports(grading_reports)
    competition_ids = load_competition_ids(split)
    grouped_competition_reports = build_complete_seeds(
        competition_reports, competition_ids, num_seeds, pad_missing=pad_missing
    )

    usable_seeds = select_seeds(grouped_competition_reports, num_seeds)

    seed_metrics: list[SeedMetrics] = []
    for i, reports in enumerate(usable_seeds):
        seed_metric = calculate_metrics_for_a_seed(reports)
        print(f"Seed {i + 1}: {seed_metric}")
        seed_metrics.append(seed_metric)

    average_metrics = average_metrics_across_seeds(seed_metrics, len(competition_ids))
    print("------\nAverage metrics across seeds:")
    print(json.dumps(asdict(average_metrics), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    exclusive = parser.add_mutually_exclusive_group(required=True)
    exclusive.add_argument(
        "--grading-reports", nargs="+", type=str, help="Paths to the grading reports"
    )
    exclusive.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment ID from runs/run_group_experiments.csv to resolve grading reports",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=-1,
        help="Number of seeds to use for averaging; use -1 to include all complete seeds",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Name or path of split file to use (default: split75).",
    )
    parser.add_argument(
        "--pad-missing",
        action="store_true",
        help=(
            "Allow seeds with missing competition reports by zero-padding the missing entries "
            "instead of discarding incomplete seeds."
        ),
    )
    args = parser.parse_args()
    reports_list: list[str]
    if args.experiment_id:
        reports_list = grading_reports_from_experiment(args.experiment_id)
    else:
        reports_list = args.grading_reports
    main(reports_list, args.num_seeds, args.split, args.pad_missing)
