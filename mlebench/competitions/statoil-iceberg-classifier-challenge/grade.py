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
    assert "is_iceberg" in answers.columns, f"Answers is missing `is_iceberg` column"

    # submission checks
    if "id" not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `id` column")
    if "is_iceberg" not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `is_iceberg` column")
    if set(submission["id"]) != set(answers["id"]):
        raise InvalidSubmissionError("Submission and answers have different id's")
    if not ((submission["is_iceberg"] >= 0) & (submission["is_iceberg"] <= 1)).all():
        raise InvalidSubmissionError(
            "All values in submission `is_iceberg` must be between 0 and 1."
        )

    # sort by id to ensure correct order
    submission = submission.sort_values("id")
    answers = answers.sort_values("id")

    y_true = answers["is_iceberg"]
    y_pred = submission["is_iceberg"]
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    score = log_loss(y_true, y_pred)
    return score
