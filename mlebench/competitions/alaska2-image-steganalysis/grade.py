import numpy as np
import pandas as pd
from sklearn import metrics

from mlebench.grade_helpers import InvalidSubmissionError


def insert_thresholds(fpr: np.ndarray, tpr: np.ndarray, tpr_thresholds: list) -> tuple:
    """
    Insert tpr_thresholds into the TPR and FPR arrays to ensure that the thresholds are present
    in the TPR array and the corresponding FPR values are interpolated.

    e.g.
    > tpr = [0.0, 0.25, 0.5, 0.75, 1.0]
    > fpr = [0.0, 0.1, 0.2, 0.3, 0.4]

    > tpr_thresholds = [0.0, 0.3, 0.4, 1.0]
    > fpr, tpr = insert_thresholds(fpr, tpr, tpr_thresholds)

    > print(tpr)
    > print(fpr)
    [0.0, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    [0.0, 0.1, 0.12, 0.16, 0.2, 0.3, 0.4]
    """
    fpr_ = fpr.tolist().copy()  # Don't modify the input arrays
    tpr_ = tpr.tolist().copy()
    for threshold in tpr_thresholds:
        if threshold not in tpr_:
            # Find the right position within tpr to insert the threshold
            for i, tpr_val in enumerate(tpr_):
                if tpr_val > threshold:
                    # Linear interpolation of fpr
                    new_fpr = fpr_[i - 1] + (fpr_[i] - fpr_[i - 1]) * (threshold - tpr_[i - 1]) / (
                        tpr_[i] - tpr_[i - 1]
                    )

                    tpr_.insert(i, threshold)
                    fpr_.insert(i, new_fpr)
                    break
    return np.array(fpr_), np.array(tpr_)


def alaska_weighted_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Instead of the standard AUC, the competition uses a weighted AUC where different regions of
    the ROC curve are weighted differently.

    We compute the area under the curve segment by segment (HORIZONTAL segments between TPR pairs
    on the y-axis), and then compute a weighted average of the segments.

    For more details, see:
    www.kaggle.com/competitions/alaska2-image-steganalysis/overview/evaluation

    This particular implementation is adapted from:
    https://www.kaggle.com/code/anokas/weighted-auc-metric-updated
    (Key change vs the above implementation: The above implementation uses the `tpr` and `fpr`
    arrays from metrics.roc_curve as is, neglecting to handle the case where the `tpr` values
    don't line up nicely with the thresholds - leading to a situation where some segments either
    belong partially to the wrong threshold, or get skipped entirely. Our implementation fixes
    this by inserting the thresholds into the `tpr` and `fpr` arrays before computing the AUC.)
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    fpr, tpr = insert_thresholds(fpr, tpr, tpr_thresholds)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    competition_metric = 0
    # Compute AUC segment by segment (where each segment is a horizontal slice of the ROC curve
    # between a pair of consecutive TPR thresholds)
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]

        # Here, we're creating new x and y arrays to calculate the AUC for this segment:

        # The segment arrays consist of the FPR and TPR values in the segment,
        mask = (y_min <= tpr) & (tpr <= y_max)
        if mask.sum() == 0:
            continue
        xs = fpr[mask]
        ys = tpr[mask]

        # plus a new point [1, y_max] which just closes the shape of this segment (draws a
        # horizontal line from the highest point in this segment to the right of the x-axis)
        xs = np.concatenate([xs, [1]])
        ys = np.concatenate([ys, [y_max]])
        ys = ys - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(xs, ys)
        submetric = score * weight
        competition_metric += submetric

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    return competition_metric / normalization


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame):
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    if "Id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an 'Id' column")

    if "Label" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'Label' column")

    submission = submission.sort_values("Id")
    answers = answers.sort_values("Id")

    if (submission["Id"].values != answers["Id"].values).any():
        raise InvalidSubmissionError("Submission and answers must have the same ids")

    # Answers and submission must be numbers
    try:
        submission["Label"] = submission["Label"].astype(float)
    except ValueError:
        raise InvalidSubmissionError("Labels in submission must be numbers")
    answers["Label"] = answers["Label"].astype(float)

    # Cannot contain NaNs
    assert not answers["Label"].isnull().any(), "Answers cannot contain NaNs"
    if submission["Label"].isnull().any():
        raise InvalidSubmissionError("Submission cannot contain NaNs")

    return answers["Label"], submission["Label"]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)

    return alaska_weighted_auc(np.array(y_true), np.array(y_pred))
