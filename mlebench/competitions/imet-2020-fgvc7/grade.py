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
        "id" in answers.columns and "attribute_ids" in answers.columns
    ), "Answers DataFrame must have 'id' and 'attribute_ids' columns"

    # Submission checks
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have an 'id' column.")
    if "attribute_ids" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'attribute_ids' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )

    # Match order
    submission = submission.sort_values(by="id").reset_index(drop=True)
    answers = answers.sort_values(by="id").reset_index(drop=True)
    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have matching 'id' columns."
        )

    # pandas reads empty cells as NaNs, which are float. We fill with empty string to match type
    submission["attribute_ids"] = submission["attribute_ids"].fillna("")
    answers["attribute_ids"] = answers["attribute_ids"].fillna("")

    # Get classes
    classes = set(answers["attribute_ids"].str.split().explode().unique())

    # Convert to sparse matrices using MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=sorted(classes), sparse_output=True)
    y_true = mlb.fit_transform(answers["attribute_ids"].str.split())
    y_pred = mlb.transform(submission["attribute_ids"].str.split())

    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")
