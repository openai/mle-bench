import numpy as np
import pandas as pd
from pandas import DataFrame

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: DataFrame, answers: DataFrame) -> dict:

    assert "Patient_Week" in answers.columns, "Answers DataFrame must have a 'Patient_Week' column."
    assert "FVC" in answers.columns, "Answers DataFrame must have a 'FVC' column."
    assert "Patient" in answers.columns, "Answers DataFrame must have a 'Patient' column."
    if "Patient_Week" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'Patient_Week' column.")
    if "FVC" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'FVC' column.")
    if "Confidence" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'Confidence' column.")
    for pw in submission["Patient_Week"]:
        if pw not in answers["Patient_Week"].values:
            raise InvalidSubmissionError(
                f"Patient_Week {pw} in submission does not exist in answers"
            )
    if not pd.api.types.is_numeric_dtype(submission["FVC"]):
        raise InvalidSubmissionError("FVC column in submission must be numeric.")
    if not pd.api.types.is_numeric_dtype(submission["Confidence"]):
        raise InvalidSubmissionError("Confidence column in submission must be numeric.")

    # fillna with 0 for the confidence column
    submission["Confidence"] = submission["Confidence"].fillna(0)

    # We should only take the 3 latest Patient_Week records for each patient
    answers = answers.dropna(subset=["FVC"])  # Drop the dummy data
    answers["Week_Number"] = answers["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
    latest_weeks = answers.sort_values("Week_Number").groupby("Patient").tail(3)
    answers = latest_weeks.drop(columns=["Week_Number"])
    # Make submission match; we only grade the prediction for the 3 latest weeks
    submission = submission[submission["Patient_Week"].isin(answers["Patient_Week"])]

    submission = submission.sort_values(by="Patient_Week")
    answers = answers.sort_values(by="Patient_Week")

    fvc_true = answers.loc[answers["Patient_Week"].isin(submission["Patient_Week"]), "FVC"].values
    fvc_pred = submission.loc[
        submission["Patient_Week"].isin(answers["Patient_Week"]), "FVC"
    ].values
    confidence = submission.loc[
        submission["Patient_Week"].isin(answers["Patient_Week"]), "Confidence"
    ].values

    return {"fvc_true": fvc_true, "fvc_pred": fvc_pred, "confidence": confidence}


def laplace_log_likelihood(
    actual_fvc: np.ndarray, predicted_fvc: np.ndarray, confidence: np.ndarray, return_values=False
) -> float:
    """
    Calculates the modified Laplace Log Likelihood score for osic-pulmonary-fibrosis-progression
    See https://www.kaggle.com/code/rohanrao/osic-understanding-laplace-log-likelihood for details on this implementation
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)


def grade(submission: DataFrame, answers: DataFrame) -> float:
    fvc_true, fvc_pred, confidence = prepare_for_metric(submission, answers).values()

    return laplace_log_likelihood(fvc_true, fvc_pred, confidence)
