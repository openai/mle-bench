import pandas as pd
from sklearn.metrics import f1_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame):
    """
    `submission` and `answers` are pd.DataFrame with "Id" and "Predicted" columns
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} != answers length {len(answers)}"
        )
    if "Id" not in submission.columns or "Predicted" not in submission.columns:
        raise InvalidSubmissionError("Submission must have 'Id' and 'Predicted' columns")

    # Sort submission and answers by "Id"
    submission = submission.sort_values("Id")
    answers = answers.sort_values("Id")
    if (submission["Id"].values != answers["Id"].values).any():
        raise InvalidSubmissionError("Submission and answers have mismatched 'Id' columns")

    y_true = [int(y) for y in answers["Predicted"]]
    y_pred = [int(y) for y in submission["Predicted"]]
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="macro")
