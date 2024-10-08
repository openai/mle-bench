import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import mean_average_precision_at_k


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    # Answers checks
    assert set(answers.columns) == {
        "customer_id",
        "prediction",
    }, "Answers DataFrame must have 'customer_id' and 'prediction' columns."

    # Submission checks
    if set(submission.columns) != {"customer_id", "prediction"}:
        raise InvalidSubmissionError(
            "Submission DataFrame must have 'customer_id' and 'prediction' columns."
        )
    if not (set(submission["customer_id"]) >= set(answers["customer_id"])):
        raise InvalidSubmissionError(
            "Submission customer_id must be a superset of answers customer_id"
        )

    # Filter the submission to only consider the customer_ids that exist in answers
    submission = submission[submission["customer_id"].isin(answers["customer_id"])]

    # Sort to ensure we're grading the right rows
    submission = submission.sort_values("customer_id").reset_index(drop=True)
    answers = answers.sort_values("customer_id").reset_index(drop=True)

    y_true = answers["prediction"].astype(str).str.split(" ").apply(set).tolist()
    y_pred = submission["prediction"].astype(str).str.split(" ").tolist()

    return {"actual": y_true, "predicted": y_pred}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Prepare the data for metric calculation
    prepped = prepare_for_metric(submission, answers)
    return mean_average_precision_at_k(
        actual=prepped["actual"], predicted=prepped["predicted"], k=12
    )
