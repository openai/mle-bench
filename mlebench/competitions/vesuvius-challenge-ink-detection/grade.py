import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import fbeta_score

from mlebench.competitions.utils import rle_decode
from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: DataFrame, answers: DataFrame) -> float:
    scores = []

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected {len(answers)} rows in the submission DataFrame, but got {len(submission)} "
            f"rows."
        )

    for (_, actual), (_, expected) in zip(submission.iterrows(), answers.iterrows()):
        score = grade_row(actual, expected)
        scores.append(score)

    return np.mean(scores)


def grade_row(submission: Series, answer: Series) -> float:
    expected_submission_indices = ["Id", "Predicted"]
    expected_answer_indices = expected_submission_indices + ["width", "height"]

    assert set(answer.index) == set(
        expected_answer_indices
    ), f"Expected indices {set(expected_answer_indices)}, got {set(answer.index)}."

    if len(submission) != len(expected_submission_indices):
        raise InvalidSubmissionError(
            f"Expected {len(expected_submission_indices)} indices in the submission DataFrame, "
            f"but got {len(submission)} indices."
        )

    if not set(submission.index).issuperset(set(expected_submission_indices)):
        raise InvalidSubmissionError(
            f"Expected columns {expected_submission_indices}, got {set(submission.index)}."
        )

    assert (
        submission["Id"] == answer["Id"]
    ), "Expected 'Id' column to be the same in both DataFrames."

    submission_rle = submission["Predicted"]
    answer_rle = answer["Predicted"]

    if not isinstance(submission_rle, str):
        raise InvalidSubmissionError(
            f"Expected 'Predicted' column to be a string, got {type(submission_rle)}."
        )

    assert isinstance(
        answer_rle, str
    ), f"Expected 'Predicted' column in answer to be a string, got {type(answer_rle)}."

    assert "width" in answer.index, "Expected 'width' index in answer."
    assert "height" in answer.index, "Expected 'height' index in answer."

    width = int(answer["width"])
    height = int(answer["height"])

    # The `rle_decode` assumes the run-length encoded string is ordered top to bottom then left to right,
    # that is, 1 is (1,1), 2 is (2,1) and so on. The Vesuvius Challenge requires the opposite order, that
    # is, 1 is (1,1), 2 is (1,2) and so on. To fix this, we decode the transpose of the matrix, then
    # transpose it back.
    try:
        submission_matrix = rle_decode(submission_rle, height=width, width=height).T
    except AssertionError as e:
        raise InvalidSubmissionError(f"Error decoding RLE masks: {e}")

    answer_matrix = rle_decode(answer_rle, height=width, width=height).T

    assert submission_matrix.shape == (
        height,
        width,
    ), f"Expected submission matrix to have shape ({height}, {width}), got {submission_matrix.shape}."

    assert (
        submission_matrix.shape == answer_matrix.shape
    ), f"Expected submission matrix to have shape {answer_matrix.shape}, got {submission_matrix.shape}."

    y_pred = submission_matrix.flatten().astype(bool)
    y_true = answer_matrix.flatten().astype(bool)

    assert np.isclose(submission_matrix.sum().sum(), y_pred.sum()), (
        f"Expected the sum of the submission matrix to be preserved when flattening and converting "
        f"to bool, but got {np.sum(y_pred)} instead of {np.sum(submission_matrix)}."
    )

    score = fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5)

    return score
