from pandas import DataFrame
from sklearn.metrics import cohen_kappa_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_grading(
    submission: DataFrame, answers: DataFrame, id_column: str, target_column: str
) -> tuple:

    # Answers checks
    assert (
        target_column in answers.columns
    ), f"Answers must contain the target column '{target_column}'"
    assert id_column in answers.columns, f"Answers must contain the id column '{id_column}'"
    assert not answers.empty, "Answers DataFrame should not be empty"

    # Submission checks
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain the target column '{target_column}'")
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain the id column '{id_column}'")
    if submission.empty:
        raise InvalidSubmissionError("Submission DataFrame should not be empty")

    # Sort the submissions and answers by id_column
    submission_sorted = submission.sort_values(by=id_column)
    answers_sorted = answers.sort_values(by=id_column)

    if (submission_sorted[id_column].values != answers_sorted[id_column].values).any():
        raise InvalidSubmissionError("Submission must contain the same ids as the answers")

    # Extract the target columns
    y_pred = submission_sorted[target_column].values
    y_true = answers_sorted[target_column].values

    return y_pred, y_true


def grade(submission: DataFrame, answers: DataFrame) -> float:
    y_pred, y_true = prepare_for_grading(
        submission=submission, answers=answers, id_column="essay_id", target_column="score"
    )
    return cohen_kappa_score(y_pred, y_true, weights="quadratic")
