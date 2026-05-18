import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame):
    """
    `submission` and `answers` are pd.DataFrame with "id" and "predicted" columns
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} != answers length {len(answers)}"
        )
    if "id" not in submission.columns or "predicted" not in submission.columns:
        raise InvalidSubmissionError("Submission must have 'id' and 'predicted' columns")

    assert "id" in answers.columns, "Answers must have 'id' column"
    assert "predicted" in answers.columns, "Answers must have 'predicted' column"

    # Sort submission and answers by "id"
    submission = submission.sort_values("id")
    answers = answers.sort_values("id")
    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Submission and answers have mismatched 'id' columns")

    y_true = [int(y) for y in answers["predicted"]]
    y_pred = [int(y) if isinstance(y, int) else int(y.split()[0]) for y in submission["predicted"]]
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return 1 - accuracy
