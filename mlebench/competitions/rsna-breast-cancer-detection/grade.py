import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError


def pfbeta(labels, predictions, beta):
    """
    Adapted from https://www.kaggle.com/code/sohier/probabilistic-f-score
    adaptations: code formatting, this docstring.
    """
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "prediction_id"
    target_col = "cancer"

    assert id_col in answers.columns, f"{id_col} not in answers"
    assert target_col in answers.columns, f"{target_col} not in answers"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"{id_col} not in submission")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"{target_col} not in submission")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("submission and answers have different lengths")

    submission = submission.sort_values(by=id_col)
    answers = answers.sort_values(by=id_col)

    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError(f"{id_col} not aligned")

    labels = answers[target_col].to_numpy()
    predictions = submission[target_col].to_numpy()

    return {"labels": labels, "predictions": predictions}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    pf1_inputs = prepare_for_metric(submission, answers)
    # F1 score, so beta=1; https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/370123
    score = pfbeta(**pf1_inputs, beta=1)
    return score
