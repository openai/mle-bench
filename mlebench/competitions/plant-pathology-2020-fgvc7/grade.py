import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    expected_answers_columns = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    expected_sample_submission_columns = expected_answers_columns

    assert set(answers.columns) == set(
        expected_answers_columns
    ), f"Expected `answers` to have columns {expected_answers_columns} but got {answers.columns}"

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected submission to have {len(answers)} rows but got {len(submission)}"
        )

    if set(submission.columns) != set(expected_sample_submission_columns):
        raise InvalidSubmissionError(
            f"Expected submission to have columns {expected_sample_submission_columns} but got {submission.columns}"
        )

    submission_sorted = submission.sort_values(by="image_id").reset_index(drop=True)
    answers_sorted = answers.sort_values(by="image_id").reset_index(drop=True)

    if not all(submission_sorted["image_id"] == answers_sorted["image_id"]):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have matching 'image_id' columns."
        )

    class_columns = ["healthy", "multiple_diseases", "rust", "scab"]
    scores = []

    for cls in class_columns:
        score = roc_auc_score(y_true=answers_sorted[cls], y_score=submission_sorted[cls])
        scores.append(score)

    return np.mean(scores)
