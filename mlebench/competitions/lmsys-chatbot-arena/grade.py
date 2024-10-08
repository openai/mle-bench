import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mlebench.competitions.utils import one_hot_dfs_to_log_loss_inputs
from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """
    The submission and answers are already one-hotted
    """
    classes = ["winner_model_a", "winner_model_b", "winner_tie"]
    required_columns = ["id"] + classes

    # Check if submission has the required columns
    missing_columns = [col for col in required_columns if col not in submission.columns]
    if missing_columns:
        raise InvalidSubmissionError(
            f"Submission DataFrame is missing required columns: {missing_columns}"
        )

    # Check if answers has the required columns
    assert set(required_columns).issubset(
        answers.columns
    ), f"Answers DataFrame is missing required columns: {set(required_columns) - set(answers.columns)}"

    # Check if submission has the correct number of rows
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission DataFrame must have {len(answers)} rows, but has {len(submission)} rows."
        )

    # Check if all values in submission are between 0 and 1
    if (
        not ((submission[classes] >= 0) & (submission[classes] <= 1)).all().all()
    ):  # first all() checks if all rows are valid, second all() checks if all columns are valid
        raise InvalidSubmissionError("All values in submission DataFrame must be between 0 and 1.")

    # Check if each row in submission sums to 1
    if not submission[classes].sum(axis=1).round(6).eq(1).all():
        raise InvalidSubmissionError("Each row in submission DataFrame must sum to 1.")

    # Use only the required columns for further processing
    submission = submission[required_columns]
    answers = answers[required_columns]

    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)

    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Submission and answer IDs do not match after sorting.")

    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission, answers, id_column="id", apply_softmax=False
    )

    return log_loss_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)
