import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import mean_average_precision_at_k


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    # Answers checks
    assert set(answers.columns) == {
        "Image",
        "Id",
    }, "Answers DataFrame must have 'Image' and 'Id' columns."

    # Submission checks
    if set(submission.columns) != {"Image", "Id"}:
        raise InvalidSubmissionError("Submission DataFrame must have 'Image' and 'Id' columns.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )

    # Ensure the IDs match
    if not set(submission["Image"]) == set(answers["Image"]):
        raise InvalidSubmissionError("Submission and answers have different Image IDs")

    # Sort to ensure we're grading the right rows
    submission = submission.sort_values("Image").reset_index(drop=True)
    answers = answers.sort_values("Image").reset_index(drop=True)

    y_true = answers["Id"].astype(str).str.split(" ").apply(set).tolist()
    y_pred = submission["Id"].astype(str).str.split(" ").tolist()

    return {"actual": y_true, "predicted": y_pred}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Prepare the data for metric calculation
    prepped = prepare_for_metric(submission, answers)
    return mean_average_precision_at_k(
        actual=prepped["actual"], predicted=prepped["predicted"], k=5
    )
