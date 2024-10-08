import numpy as np
import pandas as pd
from Levenshtein import distance

from mlebench.grade_helpers import InvalidSubmissionError


def edit_distance_array(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.mean([distance(a, b) for a, b in zip(y_true, y_pred)])


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame):
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    if "image_id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an 'image_id' column")

    if "InChI" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'InChI' column")

    assert "image_id" in answers.columns, "Answers must have 'image_id' column"
    assert "InChI" in answers.columns, "Answers must have 'InChI' column"

    submission = submission.sort_values("image_id")
    answers = answers.sort_values("image_id")

    if (submission["image_id"].values != answers["image_id"].values).any():
        raise InvalidSubmissionError("Submission and answers must have the same ids")

    # Convert to strings
    submission["InChI"] = submission["InChI"].astype(str)
    answers["InChI"] = answers["InChI"].astype(str)

    return answers["InChI"], submission["InChI"]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return edit_distance_array(y_true, y_pred)
