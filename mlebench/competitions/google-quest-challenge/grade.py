import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    id_col = "qa_id"
    classes_set = set(CLASSES)

    # submission checks
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `{id_col}` column")
    sub_cols_set = set(submission.columns)
    if not classes_set.issubset(sub_cols_set):
        raise InvalidSubmissionError(
            f"Submission is missing the following columns: {classes_set - sub_cols_set}"
        )
    if len(submission.columns) != len(CLASSES) + 1:
        raise InvalidSubmissionError(
            f"Submission has {len(submission.columns) - 1} columns, expected {len(CLASSES)}"
        )
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers have different lengths")

    # answers checks
    assert id_col in answers.columns, f"Answers is missing `{id_col}` column"
    ans_cols_set = set(answers.columns)
    assert classes_set.issubset(
        ans_cols_set
    ), f"Answers is missing the following columns: {classes_set - ans_cols_set}"
    assert (
        len(answers.columns) == len(CLASSES) + 1
    ), f"Answers has {len(answers.columns)} columns, expected {len(CLASSES) + 1}"

    # sort by id to ensure correct order
    submission = submission.sort_values(id_col)
    answers = answers.sort_values(id_col)

    spearmans = [spearmanr(submission[col], answers[col]).correlation for col in CLASSES]
    score = np.mean(spearmans)
    return score
