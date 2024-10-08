import numpy as np
from pandas import DataFrame
from sklearn.metrics import log_loss

from mlebench.competitions.utils import one_hot_dfs_to_log_loss_inputs
from mlebench.grade_helpers import InvalidSubmissionError

from .dogs import DOGS_LIST


def prepare_for_metric(submission: DataFrame, answers: DataFrame) -> dict:
    if not all(dog in submission.columns for dog in DOGS_LIST):
        raise InvalidSubmissionError(f"Submission must have columns for all dogs: {DOGS_LIST}")
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an `id` column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission should be the same length as the answers")

    assert "id" in answers.columns, "Answers must have an `id` column"
    assert all(
        dog in answers.columns for dog in DOGS_LIST
    ), f"Answers must have columns for all dogs: {DOGS_LIST}"

    tolerance = 1e-6
    if not np.all(np.isclose(submission[DOGS_LIST].sum(axis=1), 1, atol=tolerance)):
        raise InvalidSubmissionError(
            "Dog probabilities in each row in submission should sum to one, as probabilities."
        )
    if not ((submission[DOGS_LIST] >= 0) & (submission[DOGS_LIST] <= 1)).all().all():
        raise InvalidSubmissionError(
            "All probabilities in submission DataFrame must be between 0 and 1."
        )

    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission, answers, id_column="id", apply_softmax=False
    )
    return log_loss_inputs


def grade(submission: DataFrame, answers: DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)
