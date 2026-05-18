import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric
from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    roc_auc_inputs = prepare_for_auroc_metric(
        answers=answers, submission=submission, id_col="BraTS21ID", target_col="MGMT_value"
    )
    score = roc_auc_score(y_true=roc_auc_inputs["y_true"], y_score=roc_auc_inputs["y_score"])
    return score
