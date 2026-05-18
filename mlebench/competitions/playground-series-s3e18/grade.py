import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:

    if "id" not in submission.columns or "id" not in answers.columns:
        raise InvalidSubmissionError(
            "Both submission and answers DataFrames must contain an 'id' column."
        )

    if "EC1" not in submission.columns or "EC2" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must contain 'EC1' and 'EC2' columns.")

    assert (
        "EC1" in answers.columns and "EC2" in answers.columns
    ), "Answers DataFrame must contain 'EC1' and 'EC2' columns."

    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must be the same length as the answers.")

    answers_sorted = answers.sort_values("id")
    submission_sorted = submission.sort_values("id")

    if (submission_sorted["id"].values != answers_sorted["id"].values).any():
        raise InvalidSubmissionError("Submission and answers have mismatched 'id' columns")

    y_true = answers_sorted[["EC1", "EC2"]].to_numpy()
    y_score = submission_sorted[["EC1", "EC2"]].to_numpy()

    return {"y_true": y_true, "y_score": y_score}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    roc_auc_inputs = prepare_for_metric(submission, answers)
    # macro average: calc metric for each class, and then average, as outlined in comp description
    return roc_auc_score(**roc_auc_inputs, average="macro")
