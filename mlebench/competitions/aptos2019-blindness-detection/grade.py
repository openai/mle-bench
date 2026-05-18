from pandas import DataFrame
from sklearn.metrics import cohen_kappa_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_grading(submission: DataFrame, answers: DataFrame, target_column: str) -> tuple:

    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain the target column '{target_column}'")
    assert (
        target_column in answers.columns
    ), f"Answers must contain the target column '{target_column}'"
    if submission.empty:
        raise InvalidSubmissionError("Submission DataFrame should not be empty")
    assert not answers.empty, "Answers DataFrame should not be empty"
    if set(submission["id_code"]) != set(answers["id_code"]):
        raise InvalidSubmissionError("Submission must have the same id_codes as answers")

    # Ensure the submissions and answers are sorted by id_code
    submission_sorted = submission.sort_values(by="id_code")
    answers_sorted = answers.sort_values(by="id_code")

    # Extract the target columns
    y_pred = submission_sorted[target_column].values
    y_true = answers_sorted[target_column].values

    return y_pred, y_true


def grade(submission: DataFrame, answers: DataFrame) -> float:
    y_pred, y_true = prepare_for_grading(
        submission=submission, answers=answers, target_column="diagnosis"
    )
    return cohen_kappa_score(y_pred, y_true, weights="quadratic")
