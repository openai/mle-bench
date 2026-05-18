from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    # answers checks
    assert "id" in answers.columns, f"Answers is missing `id` column"
    assert "label" in answers.columns, f"Answers is missing `label` column"

    # submission checks
    if "id" not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `id` column")
    if "label" not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `label` column")
    if set(submission["id"]) != set(answers["id"]):
        raise InvalidSubmissionError("Submission and answers have different id's")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers have different lengths")
    if not ((submission["label"] >= 0) & (submission["label"] <= 1)).all():
        raise InvalidSubmissionError(
            "All values in submission `label` column must be between 0 and 1."
        )

    # sort by id to ensure correct order
    submission = submission.sort_values("id")
    answers = answers.sort_values("id")

    y_true = answers["label"]
    y_pred = submission["label"]
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    score = log_loss(y_true=y_true, y_pred=y_pred)
    return score
