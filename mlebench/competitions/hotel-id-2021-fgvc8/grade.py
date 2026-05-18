import numpy as np
import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import mean_average_precision_at_k


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "image"
    target_col = "hotel_id"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `{id_col}` column")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `{target_col}` column")
    assert id_col in answers.columns, f"Answers is missing `{id_col}` column"
    assert target_col in answers.columns, f"Answers is missing `{target_col}` column"

    if not set(submission[id_col]) == set(answers[id_col]):
        raise InvalidSubmissionError("Submission and answers have different ids")

    if not len(submission) == len(answers):
        raise InvalidSubmissionError("Submission and answers have different lengths")

    # sort to ensure we're grading the right rows
    submission = submission.sort_values(id_col).reset_index(drop=True)
    answers = answers.sort_values(id_col).reset_index(drop=True)

    # split `hotel_id` column into list of strings
    submission[target_col] = submission[target_col].astype(str).str.split(" ")
    answers[target_col] = answers[target_col].astype(str).str.split(" ").apply(set)

    actual = answers[target_col].tolist()
    predicted = submission[target_col].tolist()

    return {"actual": actual, "predicted": predicted}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    map_inputs = prepare_for_metric(submission, answers)
    return mean_average_precision_at_k(**map_inputs, k=5)
