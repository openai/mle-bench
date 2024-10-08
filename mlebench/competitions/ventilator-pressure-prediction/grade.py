from pandas import DataFrame
from sklearn.metrics import mean_absolute_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: DataFrame, answers: DataFrame) -> dict:

    assert "id" in answers.columns, "Answers must contain the 'id' column"
    assert "pressure" in answers.columns, "Answers must contain the 'pressure' column"
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'id' column")
    if "pressure" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'pressure' column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    # Ensure submission and answers are aligned in id order
    submission = submission.sort_values(by="id").reset_index(drop=True)
    answers = answers.sort_values(by="id").reset_index(drop=True)

    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Submission and answers must be aligned in id order")

    # drop any ids where answers u_out = 1 -- "The expiratory phase is not scored"
    answers = answers[answers["u_out"] == 0]
    # need to use answers to index submission
    submission = submission[submission["id"].isin(answers["id"])]

    # re-sort just to be sure
    submission = submission.sort_values(by="id").reset_index(drop=True)
    answers = answers.sort_values(by="id").reset_index(drop=True)

    return {
        "y_true": answers["pressure"].values,
        "y_pred": submission["pressure"].values,
    }


def grade(submission: DataFrame, answers: DataFrame) -> float:

    metrics = prepare_for_metric(submission=submission, answers=answers)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]

    return mean_absolute_error(y_true, y_pred)
