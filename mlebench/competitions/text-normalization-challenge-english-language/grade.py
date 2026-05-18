import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_metric(
        submission=submission,
        answers=answers,
        target_column="after",
        id_column="id",
    )
    return accuracy_score(y_pred=accuracy_inputs["y_pred"], y_true=accuracy_inputs["y_true"])


def prepare_for_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict:

    # Answers checks
    assert target_column in answers.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{id_column}` column")

    # sort on id to ensure correct order
    submission = submission.sort_values(id_column)
    answers = answers.sort_values(id_column)

    if (submission[id_column].values != answers[id_column].values).any():
        raise InvalidSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    submission[target_column] = submission[target_column].astype(str)
    answers[target_column] = answers[target_column].astype(str)

    y_pred = submission[target_column].to_numpy()
    y_true = answers[target_column].to_numpy()

    return {"y_true": y_true, "y_pred": y_pred}
