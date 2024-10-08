import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    roc_auc_inputs = prepare_for_auroc_metric(
        submission=submission, answers=answers, id_col="Id", target_col="Probability"
    )
    return roc_auc_score(y_true=roc_auc_inputs["y_true"], y_score=roc_auc_inputs["y_score"])
