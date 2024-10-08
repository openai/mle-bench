from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics

from mlebench.grade_helpers import InvalidSubmissionError

from .vocabulary import CLASS_NAMES


# Using the metric implementation given to participants. Source:
# https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8
def calculate_overall_lwlrap_sklearn(truth: np.ndarray, scores: np.ndarray) -> float:
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices],
    )
    return overall_lwlrap


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> tuple:
    """
    `submission` and `answers` are both DataFrames with N rows and M + 1 columns.
    The first column is "fname" and the remaining M columns are the predictions for each class.
    The values are floats between 0 and 1. For `answers`, the values are binary.
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    if "fname" != submission.columns[0]:
        raise InvalidSubmissionError("Submission must have an 'fname' column")

    if not all(submission.columns[1:] == CLASS_NAMES):
        raise InvalidSubmissionError("Submission classes do not match expected classes")

    assert answers.columns[0] == "fname", "Answers must have an 'fname' column"
    assert all(answers.columns[1:] == CLASS_NAMES), "Answers classes do not match expected classes"

    submission = submission.sort_values("fname").reset_index(drop=True)
    answers = answers.sort_values("fname").reset_index(drop=True)

    if (submission["fname"].values != answers["fname"].values).any():
        raise InvalidSubmissionError("Submission and answers must have the same ids")

    # Convert to numbers
    try:
        submission.iloc[:, 1:] = submission.iloc[:, 1:].apply(pd.to_numeric)
    except (ValueError, TypeError) as e:
        raise InvalidSubmissionError(f"Error converting submission values to float: {str(e)}")
    answers.iloc[:, 1:] = answers.iloc[:, 1:].apply(pd.to_numeric)

    # Return as numpy arrays
    return answers.iloc[:, 1:].values, submission.iloc[:, 1:].values


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    sub, ans = prepare_for_metric(submission, answers)
    return calculate_overall_lwlrap_sklearn(sub, ans)
