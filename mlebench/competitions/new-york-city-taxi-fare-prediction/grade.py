from pandas import DataFrame
from sklearn.metrics import root_mean_squared_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: DataFrame, answers: DataFrame) -> dict:

    assert "fare_amount" in answers.columns, "Answers should have a fare_amount column"
    assert "key" in answers.columns, "Answers should have a key column"
    if "fare_amount" not in submission.columns:
        raise InvalidSubmissionError("Submission should have a fare_amount column")
    if "key" not in submission.columns:
        raise InvalidSubmissionError("Submission should have a key column")

    # Sort by 'key' to ensure alignment
    submission = submission.sort_values("key")
    answers = answers.sort_values("key")

    y_true = answers["fare_amount"]
    y_pred = submission["fare_amount"]

    return y_true, y_pred


def grade(submission: DataFrame, answers: DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return root_mean_squared_error(y_true, y_pred)
