import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mlebench.competitions.utils import one_hot_dfs_to_log_loss_inputs
from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """
    The submission and answers are already one-hotted
    """
    if set(submission.columns) != {"id"} | set(CLASSES):
        raise InvalidSubmissionError(
            "Submission DataFrame must have an 'id' column and a column for each class."
        )
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )
    tolerance = 1e-6
    if not np.all(np.isclose(submission[CLASSES].sum(axis=1), 1, atol=tolerance)):
        raise InvalidSubmissionError("Each row in submission should sum to one, as probabilities.")
    if not ((submission[CLASSES] >= 0) & (submission[CLASSES] <= 1)).all().all():
        raise InvalidSubmissionError(
            "All probabilities in submission DataFrame must be between 0 and 1."
        )

    assert set(answers.columns) == {"id"} | set(
        CLASSES
    ), "Answers DataFrame must have an 'id' column and a column for each class."

    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission, answers, id_column="id", apply_softmax=False
    )

    return log_loss_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)
