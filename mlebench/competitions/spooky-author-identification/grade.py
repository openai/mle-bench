import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mlebench.competitions.utils import one_hot_dfs_to_log_loss_inputs
from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES

tolerance = 1e-6


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "id"

    if submission.shape != (len(answers), len(CLASSES) + 1):
        raise InvalidSubmissionError(
            f"Submission shape {submission.shape} does not match answers shape {answers.shape}."
        )

    # assert that each row in submission sums to one, accounting for floating point errors
    if not np.all(np.isclose(submission.iloc[:, 1:].sum(axis=1), 1, atol=tolerance)):
        raise InvalidSubmissionError("Each row in submission should sum to one, as probabilities.")
    if not ((submission.iloc[:, 1:] >= 0) & (submission.iloc[:, 1:] <= 1)).all().all():
        raise InvalidSubmissionError(
            "All probabilities in submission DataFrame must be between 0 and 1."
        )

    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission, answers, id_column=id_col, apply_softmax=False
    )

    return log_loss_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)
