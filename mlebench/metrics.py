import numpy as np


def average_precision_at_k(actual: set, predicted: list, k: int):
    """
    Computes the average precision at k (AP@k).

    This function computes the average precision at k between a predicted ranking and a ground truth
    set.

    Args:
        actual : A set of elements that are to be predicted (order doesn't matter)
        predicted : A list of predicted elements (order does matter, most relevant go first)
        k : The maximum number of predicted elements

    Adapted from: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mean_average_precision_at_k(actual: list[set], predicted: list[list], k: int):
    """
    Computes the MAP@k

    Args:
        actual : a list of sets of the elements that are to be predicted (order doesn't matter)
        predicted : a list of lists of predicted elements (order does matter, most relevant go first)
        k : The maximum number of predicted elements
    """
    return np.mean([average_precision_at_k(a, p, k) for a, p, in zip(actual, predicted)])


def dice_coefficient(
    predicted_mask: np.ndarray, true_mask: np.ndarray, both_empty_value: float = np.nan
) -> float:
    """
    Computes the Dice coefficient between two binary masks (can be multi-dimensional)

    Args:
        predicted_mask: A binary numpy array indicating where the segmentation is predicted
        true_mask: A binary numpy array indicating where the segmentation is
        both_empty_value: The value to return when both masks are empty
    """
    assert (
        predicted_mask.shape == true_mask.shape
    ), f"Predicted mask shape {predicted_mask.shape} does not match true mask shape {true_mask.shape}"
    # Check if both masks are empty
    if np.sum(predicted_mask) == 0 and np.sum(true_mask) == 0:
        return both_empty_value

    # Calculate intersection and union
    intersection = np.sum(predicted_mask * true_mask)
    union = np.sum(predicted_mask) + np.sum(true_mask)

    if union == 0:
        return both_empty_value

    # Calculate Dice coefficient
    dice_coeff = 2 * intersection / union

    return dice_coeff
