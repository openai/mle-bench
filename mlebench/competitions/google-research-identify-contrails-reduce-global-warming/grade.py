import numpy as np
from pandas import DataFrame

from mlebench.competitions.utils import rle_decode
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import dice_coefficient


def grade(submission: DataFrame, answers: DataFrame) -> float:
    expected_answer_columns = ["record_id", "encoded_pixels", "height", "width"]
    expected_submission_columns = ["record_id", "encoded_pixels"]

    assert set(answers.columns) == set(
        expected_answer_columns
    ), f"Expected answers to have columns {expected_answer_columns}, but got {answers.columns} instead!"

    if not set(expected_submission_columns).issubset(set(submission.columns)):
        raise InvalidSubmissionError(
            f"Expected submission to have columns {expected_submission_columns}, but got {submission.columns} instead!"
        )

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected submission to have {len(answers)} rows, but got {len(submission)} instead!"
        )

    submission = submission.sort_values(by="record_id")
    answers = answers.sort_values(by="record_id")

    if (submission["record_id"].values != answers["record_id"].values).any():
        raise InvalidSubmissionError(
            f"Expected submission to have the same record_id's as answers, but got a different set of record_id's!"
        )

    # Replace "-" in "encoded_pixels" with ""
    submission["encoded_pixels"] = submission["encoded_pixels"].replace("-", "")
    answers["encoded_pixels"] = answers["encoded_pixels"].replace("-", "")

    y_preds_rle = submission["encoded_pixels"].values
    y_trues_rle, y_true_height, y_true_width = (
        answers["encoded_pixels"].values,
        answers["height"].values,
        answers["width"].values,
    )

    try:
        y_preds = np.array(
            [
                rle_decode(rle, height=int(height), width=int(width))
                for (rle, height, width) in zip(y_trues_rle, y_true_height, y_true_width)
            ]
        )
    except AssertionError as e:
        raise InvalidSubmissionError(f"Error decoding RLE masks: {e}")

    y_trues = np.array(
        [
            rle_decode(rle, height=int(height), width=int(width))
            for (rle, height, width) in zip(y_preds_rle, y_true_height, y_true_width)
        ]
    )

    # Flatten from [3D array (list of 2D masks)] into [1D array] then compute a global dice coefficient
    """
    This competition is evaluated on the **global** Dice coefficient. The Dice coefficient can be used to compare the
    pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:
    $$
    \frac{2 \cdot |X \cap Y|}{|X| + |Y|}
    $$
    where X is the entire set of predicted contrail pixels for **all** observations in the test data and Y is the
    ground truth set of **all** contrail pixels in the test data.
    """
    y_preds_flat = y_preds.ravel()
    y_trues_flat = y_trues.ravel()
    score = dice_coefficient(y_trues_flat, y_preds_flat)

    return score
