import pandas as pd
from scipy.stats import pearsonr

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> tuple:
    """
    `submission` and `answers` are pd.DataFrames with columns `id` and `score`
    Submissions are evaluated on the Pearson correlation coefficient between the predicted and actual similarity scores.
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same number of rows")
    if not set(["id", "score"]).issubset(submission.columns):
        raise InvalidSubmissionError(
            f"Submission must have columns ['id', 'score'], got {submission.columns}"
        )

    submission = submission.sort_values("id")
    answers = answers.sort_values("id")
    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Ids mismatch between submission and answers")
    y_true = answers["score"].to_numpy()
    y_pred = submission["score"].to_numpy()

    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return pearsonr(y_true, y_pred)[0]
