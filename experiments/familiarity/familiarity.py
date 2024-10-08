import argparse
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import tiktoken
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

enc = tiktoken.encoding_for_model("gpt-4o")

# Paths relative to this file
competitions_dir = Path("../../mlebench/competitions/")
comps_list = [p.parent.name for p in sorted(competitions_dir.glob("**/description.md"))]
discussions_dir = Path("./discussions/")
meta_kaggle_dir = Path("./meta-kaggle/")
grading_reports_dir = Path("../../runs/")
test_eval_script = Path("../../tests/integration/expected_scores.py")

# import ALL_COMPS_AND_EXPECTED_SCORES from test_eval_script
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("expected_scores", test_eval_script)
expected_scores = module_from_spec(spec)
spec.loader.exec_module(expected_scores)
sample_submissions_scores = expected_scores.ALL_COMPS_AND_EXPECTED_SCORES


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(retry_if_exception_type((openai.error.APIError, IndexError))),
)
def get_prob(prompt: str, target_token: str, logprobs: int = 10) -> float | None:
    """
    Get the probability of the target token given the prompt.

    If the token is not one of the `logprobs` most probable tokens, returns None.
    """
    engine = "DUMMY-BASE-MODEL"  # Fill in with real base model engine to use
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1,
        logprobs=logprobs,
    )
    top_logprobs = response.choices[0].logprobs.top_logprobs[0]
    if target_token in top_logprobs:
        logprob = top_logprobs[target_token]
        # convert logprob to probability
        return np.exp(logprob)
    return None


def get_familiarity_score(
    text: str,
    n_samples: int,
) -> float:
    """
    Familiarity score is the mean probability of the target token over `n_samples` random places in the text.

    Example:
    ```
    test_str = "The quick brown fox jumps over the lazy dog"
    print(get_familiarity_score(test_str, n))
    test_str = "The soft-spoken purple invisible man leapt over the carton of pomegrenate juice"
    print(get_familiarity_score(test_str, n))
    ```
    """
    tokens = enc.encode(text)
    n_samples = min(n_samples, len(tokens))
    token_idxs = np.random.choice(len(tokens), n_samples, replace=False)

    total_prob = 0
    for i in tqdm(token_idxs):
        prompt_str = enc.decode(tokens[:i])
        completion_gt = enc.decode([tokens[i]])
        prob = get_prob(prompt_str, completion_gt)
        if prob is not None:
            total_prob += prob
    return total_prob / n_samples


def get_comp_metadata(comp_id: str) -> dict | None:
    # look for comp_id by matching "Slug" in experiments/regurgitation/meta-kaggle/Competitions.csv
    metadata_df = pd.read_csv(meta_kaggle_dir / "Competitions.csv")
    row = metadata_df[metadata_df["Slug"] == comp_id]
    if len(row) == 0:
        return None
    metadata = row.to_dict(orient="records")[0]
    return metadata


def get_per_comp_performance(grading_reports: list[dict]) -> dict:
    """
    Get the performance for each competition as an average across grading reports.

    Returns a dict with competition IDs as keys and the average model performance as the value
    (where performance is the % of model performance between the sample submission and the gold medal score).
    """

    performance_dict = {}
    for comp_id, sample_submission_score in sample_submissions_scores:
        if np.isnan(sample_submission_score):
            continue
        perfs = []
        for grading_report in grading_reports:
            for comp in grading_report["competition_reports"]:
                if comp["competition_id"] == comp_id:
                    if comp["score"] is None or np.isnan(comp["score"]):
                        performance = 0
                    else:
                        performance = (comp["score"] - sample_submission_score) / (
                            comp["gold_threshold"] - sample_submission_score
                        )
                    perfs.append(performance)
        if len(perfs) == 0:  # Skip if no valid performance data
            continue
        performance_dict[comp_id] = np.mean(perfs)
    return performance_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--output-cache-file",
        type=str,
        default="familiarity_and_dates.json",
        help="Cache the results to a file",
    )
    parser.add_argument(
        "--output-plot-file",
        type=str,
        default="familiarity_results.pdf",
        help="Output plot file",
    )
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the model")
    parser.add_argument(
        "--n-completions", type=int, default=1, help="Number of completions to generate"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of samples to test familiarity"
    )
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=40,
        help="Maximum number of workers for ThreadPoolExecutor",
    )
    parser.add_argument(
        "--existing-results",
        type=str,
        default=None,
        help="Use a cached results file, so we just plot the results",
    )
    args = parser.parse_args()

    familiarity_and_dates = {}

    if args.existing_results is not None:
        with open(args.existing_results, "r") as file:
            familiarity_and_dates = json.load(file)
        print(f"Reusing existing results from {args.existing_results}")

    else:

        # Loop through all "description.md" files in the competitions directory
        tasks = []
        for seed in range(args.n_seeds):
            for comp_id in comps_list:
                metadata = get_comp_metadata(comp_id)
                if metadata is None:
                    print(f"Competition {comp_id} not found in meta-kaggle, skipping")
                    continue

                tasks.append((comp_id, seed))

        def process_competition(_args):
            comp_id, seed = _args
            result = {}
            # Load the text from the file
            description_file = competitions_dir / comp_id / "description.md"
            discussions_files = list(discussions_dir.glob(f"{comp_id}/*.txt"))
            documents = [
                description_file,
                *discussions_files,
            ]
            result["familiarity_scores"] = {}
            for document in documents:
                with document.open("r", encoding="utf-8") as file:
                    text = file.read()
                score = get_familiarity_score(text, args.n_samples)
                result["familiarity_scores"][document.name] = score
            # mean familiarity score across documents
            result["mean_familiarity_score"] = sum(result["familiarity_scores"].values()) / len(
                result["familiarity_scores"]
            )

            metadata = get_comp_metadata(comp_id)
            date_str = metadata["DeadlineDate"]
            result["deadline"] = date_str
            return comp_id, seed, result

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(executor.map(process_competition, tasks))

        for comp_id, seed, result in results:
            familiarity_and_dates[f"{comp_id}_seed{seed}"] = result

        # save the scores to file
        with open(args.output_cache_file, "w") as file:
            json.dump(familiarity_and_dates, file, indent=4, sort_keys=True)

    # Get performance scores for each competition
    run_groups = []
    grading_reports = []
    for run_group in run_groups:
        run_group_dir = grading_reports_dir / run_group
        assert run_group_dir.exists(), f"Run group {run_group} not found at {run_group_dir}"
        reports = sorted(list(run_group_dir.glob("*grading_report.json")))
        assert len(reports) > 0, f"No reports found at {run_group_dir}"
        # load latest report
        with open(reports[-1], "r") as file:
            grading_reports.append(json.load(file))
    comp_performance = get_per_comp_performance(grading_reports)
    print(f"comp_performance: {comp_performance}")

    # Final set of competitions to plot
    familiarity_comp_ids = [
        comp_id.split("_seed")[0] for comp_id in list(familiarity_and_dates.keys())
    ]
    performance_comp_ids = list(comp_performance.keys())
    comp_ids = list(
        set(familiarity_comp_ids) & set(performance_comp_ids)
    )  # intersection of familiarity and performance comps
    comps_data = {}
    for comp_id in comp_ids:
        date = familiarity_and_dates[f"{comp_id}_seed0"]["deadline"]
        date = datetime.datetime.strptime(date, "%m/%d/%Y %H:%M:%S").timestamp()
        date = datetime.datetime.fromtimestamp(date)
        comps_data[comp_id] = {
            "familiarity": familiarity_and_dates[f"{comp_id}_seed0"]["mean_familiarity_score"],
            "date": date,
            "performance": comp_performance[comp_id],
        }

    # Plot: Performance vs Familiarity
    xs_familiarity = [comp_datum["familiarity"] for comp_datum in comps_data.values()]
    ys_performance = [comp_datum["performance"] for comp_datum in comps_data.values()]

    correlation, p_value = pearsonr(xs_familiarity, ys_performance)
    plt.figure(figsize=(6, 3))
    # Remove all the spines (border) around the plot
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.scatter(xs_familiarity, ys_performance)
    plt.xlabel("Familiarity", fontsize=17)
    plt.ylabel("Performance", fontsize=17)
    plt.ylim(-0.25, 1)
    plt.grid(True)
    plt.title(
        f"Pearson's correlation: {correlation:.2f}, p-value: {p_value:.2f}",
        fontsize=15,
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output_plot_file, format="pdf")
    plt.show()
