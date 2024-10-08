import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError

from . import kullback_leibler_divergence as kl_divergence
from .constants import ID_COL, TARGET_COLS


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> tuple:

    if ID_COL not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain {ID_COL} column")
    if not all(col in submission.columns for col in TARGET_COLS):
        raise InvalidSubmissionError(f"Submission must contain all target columns: {TARGET_COLS}")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")
    if not (submission[TARGET_COLS].sum(axis=1).apply(lambda x: round(x, 5) == 1).all()):
        raise InvalidSubmissionError("Submission probabilities must add to 1 for each row")
    if not set(answers[ID_COL]) == set(submission[ID_COL]):
        raise InvalidSubmissionError("Submission and answers must have the same IDs")

    assert ID_COL in answers.columns, f"Answers must contain {ID_COL} column"
    assert all(
        col in answers.columns for col in TARGET_COLS
    ), f"Answers must contain all target columns: {TARGET_COLS}"

    submission = submission.sort_values(ID_COL).reset_index(drop=True)
    answers = answers.sort_values(ID_COL).reset_index(drop=True)

    answers = answers.copy()[[ID_COL] + TARGET_COLS]
    # normalize answers to be max 1, by taking vote / sum(votes)
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468705#2606605
    answers[TARGET_COLS] = answers[TARGET_COLS].div(answers[TARGET_COLS].sum(axis=1), axis=0)

    return submission, answers


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:

    submission, answers = prepare_for_metric(submission, answers)
    score = kl_divergence.score(answers, submission, ID_COL)

    return score
