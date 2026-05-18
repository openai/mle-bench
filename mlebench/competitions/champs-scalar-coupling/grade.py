import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_grading(submission: DataFrame, answers: DataFrame, target_column: str) -> DataFrame:
    """
    Merge the submission's target_column into the answers DataFrame, matching on 'id'.
    target_column from the submission DataFrame will have the suffix '_pred',
    and target_column from the answers DataFrame will have the suffix '_true'.
    """

    # Answers checks
    assert (
        target_column in answers.columns
    ), f"Target column {target_column} not found in answers DataFrame."
    assert "id" in answers.columns, "ID column not found in answers DataFrame."
    assert "type" in answers.columns, "Type column not found in answers DataFrame."

    # Submission checks
    if target_column not in submission.columns:
        raise InvalidSubmissionError(
            f"Target column {target_column} not found in submission DataFrame."
        )
    if "id" not in submission.columns:
        raise InvalidSubmissionError("ID column not found in submission DataFrame.")
    if not set(submission["id"]) == set(answers["id"]):
        raise InvalidSubmissionError(
            f"Submission is missing the following ids: {set(answers['id']) - set(submission['id'])}"
        )

    merged = answers.merge(submission[["id", target_column]], on="id", suffixes=("_true", "_pred"))
    return merged


def log_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return np.log(mae)


def grade(submission: DataFrame, answers: DataFrame) -> float:
    """
    Submissions are evaluated on the Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types, so that a 1% decrease in MAE for one type provides the same improvement in score as a 1% decrease for another type. https://www.kaggle.com/c/champs-scalar-coupling/overview
    """
    merged = prepare_for_grading(submission, answers, "scalar_coupling_constant")

    # Group by scalar coupling type
    grouped = merged.groupby("type")

    log_maes = []
    for name, group in grouped:
        y_true = group["scalar_coupling_constant_true"]
        y_pred = group["scalar_coupling_constant_pred"]
        log_maes.append(log_mae(y_true, y_pred))

    # Average the log MAEs
    score = np.mean(log_maes)
    return score
