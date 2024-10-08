import pandas as pd
from sklearn.metrics import f1_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame):
    """
    `submission` and `answers` are pd.DataFrame with "file" and "species" columns
    """

    # Answer checks
    assert "file" in answers.columns, "Answers must have 'file' column"
    assert "species" in answers.columns, "Answers must have 'species' column"

    # Submission checks
    if "file" not in submission.columns:
        raise InvalidSubmissionError("Submission must have 'file' column")
    if "species" not in submission.columns:
        raise InvalidSubmissionError("Submission must have 'species' column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} != answers length {len(answers)}"
        )

    # Sort submission and answers by "file"
    submission = submission.sort_values("file")
    answers = answers.sort_values("file")
    if (submission["file"].values != answers["file"].values).any():
        raise InvalidSubmissionError("Submission and answers have mismatched 'file' columns")

    y_true = answers["species"].tolist()
    y_pred = submission["species"].tolist()
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")
