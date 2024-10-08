import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[csr_matrix, csr_matrix]:
    """Transforms the submission and answers DataFrames into the required format for grading as sparse matrices."""

    # Answers checks
    assert (
        "image" in answers.columns and "labels" in answers.columns
    ), "Answers DataFrame must have 'image' and 'labels' columns"

    # Submission checks
    if "image" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have an 'image' column.")
    if "labels" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'labels' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )

    # Match order
    submission = submission.sort_values(by="image").reset_index(drop=True)
    answers = answers.sort_values(by="image").reset_index(drop=True)
    if not all(submission["image"] == answers["image"]):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have matching 'image' columns."
        )

    # Get classes
    classes = set(answers["labels"].str.split().explode().unique())

    # fillna with empty string
    answers["labels"] = answers["labels"].fillna("")
    submission["labels"] = submission["labels"].fillna("")

    # Convert to sparse matrices using MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=sorted(classes), sparse_output=True)
    y_true = mlb.fit_transform(answers["labels"].str.split())
    y_pred = mlb.transform(submission["labels"].str.split())

    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")
