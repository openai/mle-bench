from typing import Tuple

import pandas as pd
from sklearn import metrics

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert "contact_id" in answers.columns, "contact_id column not found in answers"
    assert "contact" in answers.columns, "contact column not found in answers"
    if not {"contact_id", "contact"}.issubset(set(submission.columns)):
        raise InvalidSubmissionError("Submission must have columns: contact_id, contact")

    if not set(submission["contact_id"]) == set(answers["contact_id"]):
        raise InvalidSubmissionError(
            "Submission contact_id's inconsistent with answers contact_id's"
        )

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission and answers have different lengths: submission has {len(submission)} samples, answers has {len(answers)} samples"
        )

    submission = submission.sort_values("contact_id")
    answers = answers.sort_values("contact_id")

    return submission, answers


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission, answers = prepare_for_metric(submission, answers)

    return metrics.matthews_corrcoef(answers["contact"], submission["contact"])
