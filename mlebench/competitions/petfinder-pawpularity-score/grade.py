import pandas as pd
from sklearn.metrics import root_mean_squared_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "Id"
    target_col = "Pawpularity"

    assert id_col in answers.columns, f"Missing {id_col} column in answers"
    assert target_col in answers.columns, f"Missing {target_col} column in answers"
    assert (
        answers[target_col].between(1, 100).all()
    ), f"Pawpularity in answers should be between 1 and 100"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Missing {id_col} column in submission")

    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Missing {target_col} column in submission")

    if not submission[target_col].between(1, 100).all():
        raise InvalidSubmissionError(f"Pawpularity in submission should be between 1 and 100")

    submission = submission.sort_values(id_col)
    answers = answers.sort_values(id_col)

    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError(f"IDs in submission do not match IDs in answers")

    return {
        "y_true": answers[target_col].to_numpy(),
        "y_pred": submission[target_col].to_numpy(),
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    rmse_input = prepare_for_metric(submission, answers)
    score = root_mean_squared_error(**rmse_input)
    return score
