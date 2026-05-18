import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric
from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    for class_name in CLASSES:
        assert class_name in answers.columns, f"Missing class {class_name} in answers."
        if class_name not in submission.columns:
            raise InvalidSubmissionError(f"Class {class_name} is not in the submission.")
    assert len(submission) == len(
        answers
    ), f"Expected {len(answers)} rows in submission, got {len(submission)}."

    roc_auc_scores = []
    for class_name in CLASSES:
        roc_auc_inputs = prepare_for_auroc_metric(
            submission=submission,
            answers=answers,
            id_col="StudyInstanceUID",
            target_col=class_name,
        )
        roc_auc_scores.append(roc_auc_score(roc_auc_inputs["y_true"], roc_auc_inputs["y_score"]))

    score = np.average(roc_auc_scores)
    return score
