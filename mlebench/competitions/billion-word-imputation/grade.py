import numpy as np
import pandas as pd
from Levenshtein import distance

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> None:
    id_col = "id"
    target_col = "sentence"

    assert id_col in answers.columns, f"Missing {id_col} column in answers"
    assert target_col in answers.columns, f"Missing {target_col} column in answers"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Missing {id_col} column in submission")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Missing {target_col} column in submission")

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission and answers have different lengths: {len(submission)} vs {len(answers)}"
        )

    submission = submission.sort_values(id_col).reset_index(drop=True)
    answers = answers.sort_values(id_col).reset_index(drop=True)

    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError("Submission and answers should contain the same IDs")

    return submission[target_col], answers[target_col]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_pred, y_true = prepare_for_metric(submission, answers)
    mean_levenshtein = np.mean([distance(a, b) for a, b in zip(y_true, y_pred)])
    return mean_levenshtein
