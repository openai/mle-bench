import numpy as np
import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError


def jaccard(str1: str, str2: str) -> float:
    """
    Given by the competition page: https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def mean_jaccard(preds: list[str], answers: list[str]) -> float:
    return float(np.mean([jaccard(a, b) for a, b in zip(preds, answers)]))


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    The submission and answers are DataFrames with the following columns:
    ```
    textID,selected_text
    2,"very good"
    5,"I don't care"
    6,"bad"
    8,"it was, yes"
    etc.
    ```
    """

    if "textID" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'textID' column.")
    if "selected_text" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'selected_text' column.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )
    assert "textID" in answers.columns, "Answers DataFrame must have a 'textID' column."
    assert (
        "selected_text" in answers.columns
    ), "Answers DataFrame must have a 'selected_text' column."

    # Sort by textID
    submission = submission.sort_values("textID")
    answers = answers.sort_values("textID")

    if (submission["textID"].values != answers["textID"].values).any():
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same textID values."
        )

    # Note: empty strings in the submission get read as NaNs, so we fill them with empty strings
    # before converting to str (otherwise pandas will convert NaNs to "nan" strings)
    submission_labels = submission["selected_text"].fillna("").astype(str).tolist()
    answer_labels = answers["selected_text"].fillna("").astype(str).tolist()
    return submission_labels, answer_labels


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission_labels, answer_labels = prepare_for_metric(submission, answers)
    return mean_jaccard(submission_labels, answer_labels)
