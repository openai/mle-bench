# ADAPTED FROM: https://gist.github.com/SohierDane/a90ef46d79808fe3afc70c80bae45972
"""
Python equivalent of the Kuzushiji competition metric (https://www.kaggle.com/c/kuzushiji-recognition/)
Kaggle's backend uses a C# implementation of the same metric. This version is
provided for convenience only; in the event of any discrepancies the C# implementation
is the master version.

Tested on Python 3.6 with numpy 1.16.4 and pandas 0.24.2.

Update 2024/06/05: Also tested on Python 3.12 with numpy 1.26.4 and pandas 2.2.2.
"""


import multiprocessing

import numpy as np
import pandas as pd


def score_page(preds, truth):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    tp = 0
    fp = 0
    fn = 0

    truth_indices = {"label": 0, "X": 1, "Y": 2, "Width": 3, "Height": 4}
    preds_indices = {"label": 0, "X": 1, "Y": 2}

    if pd.isna(truth) and pd.isna(preds):
        return {"tp": tp, "fp": fp, "fn": fn}

    if pd.isna(truth):
        fp += len(preds.split(" ")) // len(preds_indices)
        return {"tp": tp, "fp": fp, "fn": fn}

    if pd.isna(preds):
        fn += len(truth.split(" ")) // len(truth_indices)
        return {"tp": tp, "fp": fp, "fn": fn}

    truth = truth.split(" ")
    if len(truth) % len(truth_indices) != 0:
        raise ValueError("Malformed solution string")
    truth_label = np.array(truth[truth_indices["label"] :: len(truth_indices)])
    truth_xmin = np.array(truth[truth_indices["X"] :: len(truth_indices)]).astype(float)
    truth_ymin = np.array(truth[truth_indices["Y"] :: len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + np.array(truth[truth_indices["Width"] :: len(truth_indices)]).astype(
        float
    )
    truth_ymax = truth_ymin + np.array(truth[truth_indices["Height"] :: len(truth_indices)]).astype(
        float
    )

    preds = preds.split(" ")
    if len(preds) % len(preds_indices) != 0:
        raise ValueError("Malformed prediction string")
    preds_label = np.array(preds[preds_indices["label"] :: len(preds_indices)])
    preds_x = np.array(preds[preds_indices["X"] :: len(preds_indices)]).astype(float)
    preds_y = np.array(preds[preds_indices["Y"] :: len(preds_indices)]).astype(float)
    preds_unused = np.ones(len(preds_label)).astype(bool)

    for xmin, xmax, ymin, ymax, label in zip(
        truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label
    ):
        # Matching = point inside box & character same & prediction not already used
        matching = (
            (xmin < preds_x)
            & (xmax > preds_x)
            & (ymin < preds_y)
            & (ymax > preds_y)
            & (preds_label == label)
            & preds_unused
        )
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            preds_unused[np.argmax(matching)] = False
    fp += preds_unused.sum()
    return {"tp": tp, "fp": fp, "fn": fn}


def kuzushiji_f1(sub, solution):
    """
    Calculates the competition metric.
    Args:
        sub: submissions, as a Pandas dataframe
        solution: solution, as a Pandas dataframe
    Returns:
        f1 score
    """
    if not all(sub["image_id"].values == solution["image_id"].values):
        raise ValueError("Submission image id codes don't match solution")

    pool = multiprocessing.Pool()
    results = pool.starmap(score_page, zip(sub["labels"].values, solution["labels"].values))
    pool.close()
    pool.join()

    tp = sum([x["tp"] for x in results])
    fp = sum([x["fp"] for x in results])
    fn = sum([x["fn"] for x in results])

    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision > 0 and recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1
