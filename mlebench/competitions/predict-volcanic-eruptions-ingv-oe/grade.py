from pandas import DataFrame
from sklearn.metrics import mean_absolute_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_grading(submission: DataFrame, answers: DataFrame, target_column: str) -> DataFrame:
    """
    Merge the submission's target_column into the answers DataFrame, matching on 'segment_id'.
    target_column from the submission DataFrame will have the suffix '_pred',
    and target_column from the answers DataFrame will have the suffix '_true'.
    """

    # Answers checks
    assert (
        target_column in answers.columns
    ), f"Target column {target_column} not found in answers DataFrame."
    assert "segment_id" in answers.columns, "Segment ID column not found in answers DataFrame."

    # Submission checks
    if target_column not in submission.columns:
        raise InvalidSubmissionError(
            f"Target column {target_column} not found in submissions DataFrame."
        )
    if "segment_id" not in submission.columns:
        raise InvalidSubmissionError("Segment ID column not found in submission DataFrame.")
    if not set(submission["segment_id"]) == set(answers["segment_id"]):
        raise InvalidSubmissionError(
            f"Submission is missing the following segment_ids: {set(answers['segment_id']) - set(submission['segment_id'])}"
        )

    merged = answers.merge(
        submission[["segment_id", target_column]], on="segment_id", suffixes=("_true", "_pred")
    )
    return merged


def grade(submission: DataFrame, answers: DataFrame) -> float:
    merged = prepare_for_grading(submission, answers, "time_to_eruption")
    score = mean_absolute_error(
        y_true=merged["time_to_eruption_true"], y_pred=merged["time_to_eruption_pred"]
    )
    rounded_score = int(round(score))
    return rounded_score
