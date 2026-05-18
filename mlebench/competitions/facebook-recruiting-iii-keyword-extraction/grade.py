import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.utils import get_logger

logger = get_logger(__name__)


def grade(submission: DataFrame, answers: DataFrame) -> float:
    """Grades the submission against the test set."""

    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[csr_matrix, csr_matrix]:

    # Answer checks
    assert "Id" in answers.columns, "Answers must have an 'Id' column."
    assert "Tags" in answers.columns, "Answers must have a 'Tags' column."

    # Submission checks
    if "Id" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have an 'Id' column.")
    if "Tags" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'Tags' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission DataFrame must have the same number of rows as the answers DataFrame."
        )

    # Match order
    submission = submission.sort_values(by="Id").reset_index(drop=True)
    answers = answers.sort_values(by="Id").reset_index(drop=True)
    if (submission["Id"].values != answers["Id"].values).any():
        raise InvalidSubmissionError("Submission and answers must have matching 'Id's.")

    # Get classes
    classes = set()

    for tags in answers["Tags"]:
        if not isinstance(tags, str):
            logger.warning(f"Tags from answers '{tags}' not of type str! Skipping.")
            continue

        tags_split = tags.split()
        classes.update(tags_split)

    # Convert to sparse matrix using MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=sorted(classes), sparse_output=True)
    y_true = mlb.fit_transform(answers["Tags"].fillna("").str.split())
    y_pred = mlb.transform(submission["Tags"].fillna("").str.split())

    return y_true, y_pred
