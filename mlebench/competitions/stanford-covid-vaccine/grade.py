import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected submission to be the same length as answers, but got {len(submission)} "
            f"instead of {len(answers)}."
        )

    to_predict = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    expected_answer_columns = ["id_seqpos"] + to_predict + ["keep"]

    assert set(answers.columns).issuperset(expected_answer_columns), (
        f"Expected answers to have columns {expected_answer_columns}, but instead it has "
        f"columns {answers.columns}."
    )

    # The submission csv contains two columns which aren't used for scoring: `deg_pH10` and
    # `deg_50C`. These are nonetheless still required to be in the submission as per the
    # competition rules. See the "Sample Submission" section of the competition overview page for
    # more information:
    # https://www.kaggle.com/competitions/stanford-covid-vaccine/overview/evaluation
    expected_submission_columns = ["id_seqpos"] + to_predict + ["deg_pH10", "deg_50C"]

    if not set(submission.columns).issuperset(expected_submission_columns):
        raise InvalidSubmissionError(
            f"Expected the submission to have columns {expected_submission_columns}, but instead "
            f"it has columns {submission.columns}."
        )

    filtered_submission = submission[expected_submission_columns]

    # Sort rows by `id_seqpos` and columns alphabetically
    sorted_submission = filtered_submission.sort_values(by="id_seqpos").sort_index(axis=1)
    sorted_answers = answers.sort_values(by="id_seqpos").sort_index(axis=1)

    for i, (actual_id, expected_id) in enumerate(
        zip(sorted_submission["id_seqpos"], sorted_answers["id_seqpos"])
    ):
        if actual_id == expected_id:
            continue

        raise InvalidSubmissionError(
            f"Expected submission to have the same `id_seqpos` as answers, but got `{actual_id}` "
            f"instead of `{expected_id}` on row {i} of the submission."
        )

    mask = sorted_answers["keep"]
    new_submission = sorted_submission[mask]
    new_answers = sorted_answers[mask]

    errors = []

    for column in to_predict:
        y_pred = new_submission[column]
        y_true = new_answers[column]
        error = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        errors.append(error)

    return np.mean(errors)
