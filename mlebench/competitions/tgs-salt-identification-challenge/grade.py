import numpy as np
from numpy import ndarray
from pandas import DataFrame

from mlebench.competitions.utils import rle_decode
from mlebench.grade_helpers import InvalidSubmissionError

IMG_HEIGHT = 101
IMG_WIDTH = 101


def grade(submission: DataFrame, answers: DataFrame) -> float:
    expected_columns = ["id", "rle_mask"]

    assert set(answers.columns) == set(
        expected_columns
    ), f"Expected answers to have columns {expected_columns}, but got {answers.columns} instead!"

    if set(submission.columns) != set(expected_columns):
        raise InvalidSubmissionError(
            f"Expected submission to have columns {expected_columns}, but got {submission.columns} instead!"
        )

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected submission to have {len(answers)} rows, but got {len(submission)} instead!"
        )

    transformed_submission = submission.fillna("")
    transformed_answers = answers.fillna("")

    assert transformed_answers.applymap(
        lambda x: isinstance(x, str)
    ).values.all(), "Expected all items in `answers` to be strings, but found non-string items!"

    if not transformed_submission.applymap(lambda x: isinstance(x, str)).values.all():
        raise InvalidSubmissionError(
            "Expected all items in `submission` to be strings, but found non-string items!"
        )

    sorted_submission = transformed_submission.sort_values(by="id")
    sorted_answers = transformed_answers.sort_values(by="id")

    if (sorted_submission["id"].values != sorted_answers["id"].values).any():
        raise InvalidSubmissionError(
            f"Expected submission to have the same IDs as answers, but got a different set of IDs!"
        )

    y_preds_rle = sorted_submission["rle_mask"].values
    y_trues_rle = sorted_answers["rle_mask"].values

    try:
        y_preds = np.array(
            [rle_decode(rle, height=IMG_HEIGHT, width=IMG_WIDTH) for rle in y_preds_rle]
        )
    except AssertionError as e:
        raise InvalidSubmissionError(f"Error decoding RLE masks: {e}")

    y_trues = np.array([rle_decode(rle, height=IMG_HEIGHT, width=IMG_WIDTH) for rle in y_trues_rle])

    scores = [iou_metric(y_trues[i], y_preds[i]) for i in range(len(y_trues))]
    score = np.mean(scores)

    return score


def iou_metric(y_true_in: ndarray, y_pred_in: ndarray) -> float:
    """
    Adapted from https://www.kaggle.com/code/phoenigs/u-net-dropout-augmentation-stratification.
    """

    if np.sum(y_true_in) == 0 and np.sum(y_pred_in) == 0:
        return 1.0

    if np.sum(y_true_in) == 0 and np.sum(y_pred_in) > 0:
        return 0.0

    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        labels.flatten(),
        y_pred.flatten(),
        bins=(true_objects, pred_objects),
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)

        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0

        prec.append(p)

    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    """
    Adapted from https://www.kaggle.com/code/phoenigs/u-net-dropout-augmentation-stratification.
    """

    batch_size = y_true_in.shape[0]
    metric = []

    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)

    return np.mean(metric)
