import numpy as np
import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError

column_order = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def unmelt_df(melted_df: pd.DataFrame, id_col: str, target_col: str) -> pd.DataFrame:
    unmelted_df = melted_df.copy()
    unmelted_df[["StudyInstanceUID", "variable"]] = unmelted_df[id_col].str.split(
        "_", n=1, expand=True
    )
    unmelted_df = unmelted_df.pivot(
        index="StudyInstanceUID", columns="variable", values=target_col
    ).reset_index()
    return unmelted_df


def binary_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes binary log loss for each sample and variable (no reduction).

    Args:
        y_true: True binary labels, shape (n_samples, n_variables)
        y_pred: Predicted probabilities, shape (n_samples, n_variables)

    Returns:
        Binary log loss for each sample and variable, shape (n_samples, n_variables)
    """
    epsilon = 1e-15
    # avoid y_preds of 0 or 1, since we'll be taking the log and that is undefined/inf
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def custom_comp_weighted_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Adapted from
    https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854

    Adaptations:
        - custom binary_log_loss function in numpy to avoid torch dependency
        - working in numpy arrays instead of torch tensors
        - explanatory comments
        - added comp weights inside fn
    """
    # https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392
    # assuming ordering [patient_overall, C1, C2, C3, C4, C5, C6, C7]
    competition_weights = {
        # weights for negative labels
        "-": np.array([7, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        # weights for positive labels
        "+": np.array([14, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32),
    }

    # (n_samples, n_variables)
    loss = binary_log_loss(y_true, y_pred)
    # (n_samples, n_variables) weights matrix, conditionally built based on y_true values
    weights = y_true * competition_weights["+"] + (1 - y_true) * competition_weights["-"]
    # (n_samples, n_variables) element-wise multiplication of loss and weights
    loss = loss * weights
    # (n_samples,) sum of weighted losses across variables
    loss = np.sum(loss, axis=1)
    # (n_samples,) normalize the loss by the sum of weights
    loss = loss / weights.sum(axis=1)
    # finally, take the mean across all samples
    return np.mean(loss)


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    id_col = "row_id"
    target_col = "fractured"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing the `{id_col}` column.")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing the `{target_col}` column.")
    if not submission[target_col].between(0, 1).all():
        raise InvalidSubmissionError(f"Submission `{target_col}` values must be between 0 and 1.")
    if submission[target_col].isnull().any():
        raise InvalidSubmissionError(f"Submission `{target_col}` values must not be missing.")

    assert id_col in answers.columns, f"Answers are missing the `{id_col}` column."
    assert target_col in answers.columns, f"Answers are missing the `{target_col}` column."

    # all answers should be either 0 or 1
    assert answers[target_col].isin([0, 1]).all(), f"Answers `{target_col}` values must be 0 or 1."
    # all answers should be present
    assert (
        not answers[target_col].isnull().any()
    ), f"Answers `{target_col}` values must not be missing."

    # sort both submission and answers by id_col
    submission = submission.sort_values(by=id_col)
    answers = answers.sort_values(by=id_col)

    # check that the ids match
    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError("Submission should contain same ids as answers.")

    # checks complete; now we can prepare

    # need to unmelt both submission and answers, back to getting one row per StudyInstanceUID
    unmelted_submission = unmelt_df(melted_df=submission, id_col=id_col, target_col=target_col)
    unmelted_answers = unmelt_df(melted_df=answers, id_col=id_col, target_col=target_col)

    # sort both by StudyInstanceUID
    unmelted_submission = unmelted_submission.sort_values(by="StudyInstanceUID")
    unmelted_answers = unmelted_answers.sort_values(by="StudyInstanceUID")

    # extract the target columns
    y_true = unmelted_answers[column_order].to_numpy()
    y_pred = unmelted_submission[column_order].to_numpy()

    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return custom_comp_weighted_loss(y_true, y_pred)
