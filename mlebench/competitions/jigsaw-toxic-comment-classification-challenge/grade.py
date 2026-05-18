import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    classes_set = set(CLASSES)
    id_col = "id"

    # submission checks
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing the column: {id_col}")
    sub_cols_set = set(submission.columns)
    if not classes_set.issubset(sub_cols_set):
        raise InvalidSubmissionError(
            f"Submission is missing the following columns: {classes_set - sub_cols_set}"
        )

    # answers checks
    assert id_col in answers.columns, f"Answers is missing `{id_col}` column"
    ans_cols_set = set(answers.columns)
    assert classes_set.issubset(
        ans_cols_set
    ), f"Answers is missing the following columns: {classes_set - ans_cols_set}"
    assert len(submission) == len(answers), "Submission and answers have different lengths"

    submission = submission.set_index("id").sort_index()
    answers = answers.set_index("id").sort_index()

    # skip rows marked with -1 in y_true
    # when it happens entire row is marked so we can check negative sum of the row
    keep_mask = answers[CLASSES].sum(axis=1) >= 0
    answers = answers[keep_mask]
    submission = submission[keep_mask]

    roc_auc_inputs = {
        "y_score": submission.to_numpy(),
        "y_true": answers.to_numpy(),
        # metric for each column, then average across columns
        "average": "macro",
    }

    return roc_auc_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Computes the column-wise mean ROC AUC score for the submission.
    """
    roc_auc_inputs = prepare_for_metric(submission, answers)
    return roc_auc_score(**roc_auc_inputs)
