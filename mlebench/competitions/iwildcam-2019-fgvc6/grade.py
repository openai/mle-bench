import pandas as pd
from sklearn.metrics import f1_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "Id",
    target_col: str = "Category",
):
    """
    `submission` and `answers` are pd.DataFrame with `id_col` and `target_col` columns
    """

    # Answers checks
    assert id_col in answers.columns, f"Answers must have '{id_col}' column"
    assert target_col in answers.columns, f"Answers must have '{target_col}' column"

    # Submission checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} != answers length {len(answers)}"
        )
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{id_col}' column")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{target_col}' column")

    # Sort submission and answers by "Id"
    submission = submission.sort_values(id_col)
    answers = answers.sort_values(id_col)
    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError(f"Submission and answers have mismatched '{id_col}' columns")

    y_true = [int(y) for y in answers[target_col]]
    y_pred = [int(y) for y in submission[target_col]]
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="macro")
