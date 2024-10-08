from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mlebench.grade_helpers import InvalidSubmissionError


def calculate_iou(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    (this image helps: https://machinelearningspace.com/wp-content/uploads/2023/01/IOU2.jpg)
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def calculate_map(submission_preds: List[List[Tuple]], answers_preds: List[List[Tuple]]) -> float:
    """
    Calculate mean Average Precision (mAP) for object detection.

    >   The challenge uses the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.5.
        Note that the linked document describes VOC 2012, which differs in some minor ways
        (e.g. there is no concept of "difficult" classes in VOC 2010).
        The P/R curve and AP calculations remain the same.
    (www.kaggle.com/competitions/siim-covid19-detection/overview/evaluation)

    Some choices made here that were not explicitly specified in the challenge description:

    1. Treating "none" or "negative" as prediction classes of their own, instead of as the non-positive class
    -   Justification: Treating them as their own classes is implied by the data format:
        - Study level - train_study_level.csv has 4 binary classes including the "negative" class instead of 3
        - Image level - We're asked to predict "none" with full bounding boxes instead of withholding a prediction
        Also, 3rd place winner says "It probably treats the six classes equally", where the six classes are
        "negative", "typical", "indeterminate", "atypical", "none" and "opacity".
        (https://www.kaggle.com/competitions/siim-covid19-detection/discussion/240363)

    2. Rules for populating y_pairs (see comments below), in particular the (0, 0) case
    -   Justification: The general rules follow the descriptions of the PASCAL VOC 2010 mAP documented online.
        The only custom addition is handling the edge case of (0, 0) false negatives, which was necessary because
        if we don't include (0, 0) pairs, both the sample submission and gold submission end up with all values of
        y_true being 1, so the AP is undefined.
        Behavior of our implementation is consistent with this comment from the organizers:
        https://www.kaggle.com/competitions/siim-covid19-detection/discussion/248467#1362916

    """
    aps = []

    # Group predictions by class - the general idea is to calculate AP for each class separately
    # and then average them to get mAP
    classes = sorted(list(set(pred[0] for preds in answers_preds for pred in preds)))

    for cls in classes:
        # We will populate y_true and y_scores with (y_true, y_score) of:
        # (1, confidence) for every predicted box that matches a ground truth box
        # (0, confidence) for every predicted box that does not match a ground truth box
        # (1, 0) for every ground truth box that is not matched to a predicted box
        # (0, 0) when there are neither predicted nor ground truth boxes
        y_pairs = []  # List of (y_true, y_score) pairs

        # Gather all predictions and ground truth boxes related to this class from all samples
        for img_preds, img_gts in zip(submission_preds, answers_preds):
            y_pairs_ = []

            # Get ground truth boxes for this class
            gt_boxes = [gt[2:] for gt in img_gts if gt[0] == cls]

            # Sort img_preds by confidence
            img_preds.sort(key=lambda x: x[1], reverse=True)

            # For each prediction of this class
            matched_gt = [False] * len(gt_boxes)  # Initialize all ground truths as unmatched
            for pred in img_preds:
                if pred[0] == cls:
                    # Find the best matching ground truth box
                    best_iou = 0
                    best_gt_idx = -1
                    for i, gt in enumerate(gt_boxes):
                        if matched_gt[i]:  # Don't reuse matched ground truths
                            continue
                        iou = calculate_iou(pred[2:], gt)
                        if iou > best_iou and iou > 0.5:
                            best_iou = iou
                            best_gt_idx = i

                    pred_confidence = pred[1]
                    if best_gt_idx != -1:
                        y_pairs_.append((1, pred_confidence))  # True positive
                        matched_gt[best_gt_idx] = True
                    else:
                        y_pairs_.append((0, pred_confidence))  # False positive

            # Add false negatives for unmatched ground truths
            y_pairs_.extend([(1, 0)] * matched_gt.count(False))

            if len(y_pairs_) == 0:
                # A true negative
                y_pairs_.append((0, 0))

            y_pairs.extend(y_pairs_)

        y_true = [pair[0] for pair in y_pairs]
        y_scores = [pair[1] for pair in y_pairs]
        if len(y_true) > 0:
            assert not all(
                y == 1 for y in y_true
            ), "y_true is all 1s; this shouldn't happen and will result in undefined AP"
            ap = average_precision_score(y_true, y_scores)
            aps.append(ap)
        else:
            raise ValueError(f"Unexpected error: y_true is empty for class {cls}")

    # Calculate mAP
    return np.mean(aps) if aps else 0.0


def parse_prediction_string(
    prediction_string: str,
) -> List[Tuple[str, float, float, float, float, float]]:
    """
    `prediction_string` should have the form "{label} {confidence} {xmin} {ymin} {xmax} {ymax}" for each bounding box
    (so there may be 6 x N space-separated values in total, where N is the number of bounding boxes).

    Returns a list of tuples, each with the form (label, confidence, xmin, ymin, xmax, ymax).
    """
    toks = prediction_string.strip().split()
    try:
        return [
            (
                toks[i],
                float(toks[i + 1]),
                float(toks[i + 2]),
                float(toks[i + 3]),
                float(toks[i + 4]),
                float(toks[i + 5]),
            )
            for i in range(0, len(toks), 6)
        ]
    except ValueError as e:
        raise InvalidSubmissionError(f"Could not parse PredictionString: {e}")


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade the submission against the answers.

    `submission` and `answers` are DataFrames with the following columns:
    - `id` (str): The study ID
    - `PredictionString` (str): Has the form "{label} {confidence} {xmin} {ymin} {xmax} {ymax}" for each bounding box
      (so there may be 6 x N space-separated values in total, where N is the number of bounding boxes).
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an 'Id' column")

    if "PredictionString" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'PredictionString' column")

    assert "id" in answers.columns, "Answers must have an 'id' column"
    assert "PredictionString" in answers.columns, "Answers must have a 'PredictionString' column"

    submission = submission.sort_values("id")
    answers = answers.sort_values("id")
    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Submission and answers must have the same ids")

    # Attempt to parse the PredictionString
    submission_preds = submission["PredictionString"].apply(parse_prediction_string)
    answers_preds = answers["PredictionString"].apply(parse_prediction_string)

    # Calculate the mAP
    return calculate_map(submission_preds, answers_preds) if len(submission_preds) > 0 else 0.0
