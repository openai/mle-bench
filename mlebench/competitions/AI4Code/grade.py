from bisect import bisect
from typing import Any

import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError


def kendall_tau(ground_truth: list[list[Any]], predictions: list[list[Any]]) -> float:
    """
    Computes the Kendall Tau correlation between
    - `ground_truth`: a list of lists, where each sublist contains the ground truth order of items
    - `predictions`: a list of lists, where each sublist contains the predicted order of items

    See: https://www.kaggle.com/code/ryanholbrook/competition-metric-kendall-tau-correlation
    """

    # Actually O(N^2), but fast in practice for our data
    def count_inversions(a):
        inversions = 0
        sorted_so_far = []
        for i, u in enumerate(a):  # O(N)
            j = bisect(sorted_so_far, u)  # O(log N)
            inversions += i - j
            sorted_so_far.insert(j, u)  # O(N)
        return inversions

    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")
    if "id" not in submission.columns or "cell_order" not in submission.columns:
        raise InvalidSubmissionError("Submission must have 'id' and 'cell_order' columns")
    if set(submission["id"]) != set(answers["id"]):
        raise InvalidSubmissionError("Submission and answers must have the same ids")
    # "cell_order" columns must be strings of space-separated cell ids: "54c7cab3 fe66203e 7844d5f8 ..."
    submission["cell_order"] = submission["cell_order"].astype(str)
    answers["cell_order"] = answers["cell_order"].astype(str)

    # Sort by id to ensure we're comparing the same rows
    submission = submission.sort_values("id")
    answers = answers.sort_values("id")

    ground_truth, predictions = [], []
    target_column = "cell_order"
    for (_, row_gt), (_, row_pred) in zip(submission.iterrows(), answers.iterrows()):
        ground_truth_items = row_gt[target_column].split()
        prediction_items = row_pred[target_column].split()
        if len(ground_truth_items) != len(prediction_items):
            raise InvalidSubmissionError("Ground truth and predictions must have the same length")
        ground_truth.append(ground_truth_items)
        predictions.append(prediction_items)
    return {
        "ground_truth": ground_truth,
        "predictions": predictions,
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    kendall_tau_inputs = prepare_for_metric(submission, answers)
    return kendall_tau(**kendall_tau_inputs)
