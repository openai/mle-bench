from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import softmax

from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.utils import get_logger

logger = get_logger(__name__)


def df_to_one_hot(
    df: pd.DataFrame, id_column: str, target_column: str, classes: list[str]
) -> pd.DataFrame:
    """
    Convert class labels to one-hot encoded vectors.
    """

    y_onehot = pd.DataFrame(0, index=df.index, columns=[id_column] + classes)
    y_onehot[id_column] = df[id_column]

    for i, row in df.iterrows():
        y_onehot.loc[i, row[target_column]] = 1

    return y_onehot


def one_hot_dfs_to_log_loss_inputs(
    submission_one_hot: pd.DataFrame,
    answers_one_hot: pd.DataFrame,
    id_column: str = "id",
    apply_softmax: bool = True,
) -> dict:
    """
    Frequently used logic to prepare one-hotted dfs for log loss calculation.
    """
    required_cols = set(answers_one_hot.columns)
    submission_cols = set(submission_one_hot.columns)

    if not submission_cols.issuperset(required_cols):
        raise InvalidSubmissionError(
            f"The submission DataFrame is missing some columns required by the `answers` DataFrame. "
            f"Missing columns: {required_cols - submission_cols}."
        )
    if id_column not in submission_one_hot.columns:
        raise InvalidSubmissionError(f"Submission is missing id column '{id_column}'.")

    assert id_column in answers_one_hot.columns, f"Answers is missing id column '{id_column}'."

    # Filter submission to only include columns that are in the answers
    submission_filtered = submission_one_hot[
        [col for col in answers_one_hot.columns if col in submission_cols]
    ]

    # Sort submission and answers by id to align them
    submission_sorted = submission_filtered.sort_values(by=id_column).reset_index(drop=True)
    answers_sorted = answers_one_hot.sort_values(by=id_column).reset_index(drop=True)

    assert submission_sorted[id_column].tolist() == answers_sorted[id_column].tolist(), (
        f"Mismatch in {id_column.capitalize()}s between `submission` and `answers` after sorting. "
        f"Number of mismatched {id_column.capitalize()}s: {len(set(submission_sorted[id_column]) ^ set(answers_sorted[id_column]))}. "
        f"Ensure both DataFrames have the same {id_column.capitalize()}s."
    )

    assert list(submission_sorted.columns) == list(answers_sorted.columns), (
        "Column order mismatch after filtering and sorting. "
        "Ensure both DataFrames have columns in the same order."
    )

    y_true = answers_sorted.drop(columns=[id_column]).to_numpy()
    y_pred = submission_sorted.drop(columns=[id_column]).to_numpy()

    if apply_softmax and is_one_hot_encoded(y_pred):
        logger.warning(
            "The flag `apply_softmax` has been set to `True` but the submission is already "
            "one-hot encoded. Skipping softmax."
        )

    if apply_softmax and not is_one_hot_encoded(y_pred):
        y_pred = softmax(y_pred, axis=-1)

    log_loss_inputs = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return log_loss_inputs


def is_one_hot_encoded(xs: ndarray) -> bool:
    """Check if a 2D NumPy array is one-hot encoded."""

    assert isinstance(xs, ndarray), f"Expected a NumPy array, got {type(xs)}."
    assert xs.ndim == 2, f"Expected a 2D array, got {xs.ndim}D."

    is_binary_matrix = np.bitwise_or(xs == 0, xs == 1).all()
    is_normalized = np.allclose(xs.sum(axis=-1), 1)
    is_one_hot = bool(is_binary_matrix and is_normalized)

    assert isinstance(is_one_hot, bool), f"Expected a boolean, got {type(is_one_hot)}."

    return is_one_hot


def rle_decode(rle_string: str, height: int, width: int) -> ndarray:
    """
    Decode an RLE string into a binary mask. The RLE encoding is top-down, left-right. So 1 is
    (1,1), 2 is (2, 1), etc. The RLE is 1-indexed. Checks that the pairs are sorted, positive, and
    the decoded pixel values do not overlap.

    Args:
        rle_string (str): The RLE string.
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        np.array: The decoded binary mask.
    """

    assert isinstance(
        rle_string, str
    ), f"Expected a string, but got {type(rle_string)}: {rle_string}"
    assert isinstance(height, int), f"Expected an integer, but got {type(height)}: {height}"
    assert isinstance(width, int), f"Expected an integer, but got {type(width)}: {width}"

    if not rle_string.strip():  # Check if the string is empty or contains only whitespace
        return np.zeros((height, width), dtype=bool)

    s = list(map(int, rle_string.split()))
    starts, lengths = s[0::2], s[1::2]

    assert starts == sorted(starts), "The pairs in the RLE string must be sorted."
    assert all(x > 0 for x in starts), "All pairs in the RLE string must be positive integers."
    assert all(x > 0 for x in lengths), "All pairs in the RLE string must be positive integers."

    # Convert to 0-based indices
    starts = np.array(starts) - 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=bool)

    for lo, hi in zip(starts, ends):
        assert not img[lo:hi].any(), "Overlapping RLE pairs are not allowed."
        img[lo:hi] = True

    # reshape appropriately given how the RLE was encoded
    return img.reshape((width, height)).T


# https://www.kaggle.com/code/inversion/contrails-rle-submission
def rle_encode(x: ndarray, fg_val=1):
    """
    Args:
        x:  numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encoding as list
    """
    dots = np.where(x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rles_to_masks(
    rl_encodings: list[str], image_heights: list[int], image_widths: list[int]
) -> list[np.ndarray]:
    """
    Performs run-length decoding on a list of run-length encodings to get the binary masks
    """
    masks = [
        rle_decode(encoding, height=image_height, width=image_width)
        for encoding, image_height, image_width in zip(rl_encodings, image_heights, image_widths)
    ]
    return masks


def get_ids_from_tf_records(tf_record_path: Path, id_feature: str = "image_name") -> list[str]:
    import tensorflow as tf  # Import only if needed, otherwise it slows down the module import

    tf_record_dataset = tf.data.TFRecordDataset(tf_record_path.as_posix())

    ids = []
    for record in tf_record_dataset:
        features = tf.train.Example.FromString(record.numpy())
        id = features.features.feature[id_feature].bytes_list.value[0].decode("utf-8")
        ids.append(id)

    return ids


def prepare_for_accuracy_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict:

    # Answers checks
    assert target_column in answers.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{id_column}` column")

    # sort on id to ensure correct order
    submission = submission.sort_values(id_column)
    answers = answers.sort_values(id_column)

    if (submission[id_column].values != answers[id_column].values).any():
        raise InvalidSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    y_pred = submission[target_column].to_numpy()
    y_true = answers[target_column].to_numpy()

    return {"y_true": y_true, "y_pred": y_pred}


def prepare_for_auroc_metric(
    submission: pd.DataFrame, answers: pd.DataFrame, id_col: str, target_col: str
) -> dict:

    # Answers checks
    assert id_col in answers.columns, f"answers dataframe should have an {id_col} column"
    assert target_col in answers.columns, f"answers dataframe should have a {target_col} column"

    # Submission checks
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission should have an {id_col} column")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission should have a {target_col} column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Submission and answers should have the same number of rows")
    try:
        pd.to_numeric(submission[target_col])
    except ValueError:
        raise InvalidSubmissionError(
            f"Expected {target_col} column to be numeric, got {submission[target_col].dtype} instead"
        )
    if submission[target_col].min() < 0 or submission[target_col].max() > 1:
        raise InvalidSubmissionError(
            f"Submission {target_col} column should contain probabilities,"
            " and therefore contain values between 0 and 1 inclusive"
        )
    # Sort
    submission = submission.sort_values(id_col)
    answers = answers.sort_values(id_col)

    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError(f"Submission and answers should have the same {id_col} values")

    roc_auc_inputs = {
        "y_true": answers[target_col].to_numpy(),
        "y_score": submission[target_col].to_numpy(),
    }

    return roc_auc_inputs
