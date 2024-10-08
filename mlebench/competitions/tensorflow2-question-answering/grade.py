import pandas as pd
from sklearn.metrics import f1_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: dict):
    """
    `submission` is a pd.DataFrame with "example_id" and "PredictionString" columns.

    `answers` is a dict in the format of "simplified-nq-train.jsonl" described here:
    https://www.kaggle.com/c/tensorflow2-question-answering/data

    The competition uses a micro F1 score metric, which is a binary classification metric.
    For retrieval, we convert the submission and answers for each sample as:
    - y_pred: if the submission has an answer, it is 1, else 0
    - y_true:
        - if the submission has an answer and it exists in the true labels, y_true=1
        - if the submission has an answer but it does not exist in the true labels, y_true=0
        - if the submission has no answer but there exists a true label, y_true=1
        - if the submission has no answer and there is no true label, y_true=0

    This is consistent with
    https://www.kaggle.com/competitions/tensorflow2-question-answering/overview/evaluation
    - TP = the predicted indices match one of the possible ground truth indices
    - FP = the predicted indices do NOT match one of the possible ground truth indices, OR
        a prediction has been made where no ground truth exists
    - FN = no prediction has been made where a ground truth exists

    Returns y_true, y_pred which are lists of 0s and 1s.
    """
    if len(submission) != 2 * len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} != 2 * answers length {len(answers)}"
        )
    # Empty strings are read as NaN by pandas, but we want these to remain empty strings
    submission.fillna("", inplace=True)
    submission = submission.astype(str)

    expected_ids = []
    for sample in answers:
        expected_ids.append(f"{sample['example_id']}_long")
        expected_ids.append(f"{sample['example_id']}_short")

    if not ({"example_id", "PredictionString"}).issubset(submission.columns):
        raise InvalidSubmissionError(
            "Submission requires 'example_id' and 'PredictionString' columns."
        )
    if not set(submission["example_id"]) == set(expected_ids):
        raise InvalidSubmissionError(
            "Submission example_ids do not match expected_ids. Please ensure you have a "
            "long and short answer for each example_id in the answers."
        )
    if not len(set(submission["example_id"])) == len(submission):
        raise InvalidSubmissionError(
            "Submission example_ids are not unique. Please ensure you have a "
            "long and short answer for each example_id in the answers."
        )

    y_pred = [1 if row["PredictionString"].strip() else 0 for _, row in submission.iterrows()]

    y_true = []
    true_annotations = {sample["example_id"]: sample["annotations"][0] for sample in answers}
    for idx, row in submission.iterrows():
        sample_id = row["example_id"]

        # Parse prediction
        # Prediction may be any of "{start_token}:{end_token}", "YES", "NO", ""
        pred = row["PredictionString"].strip()
        if ":" in pred and len(pred.split(":")) == 2:
            start_token, end_token = pred.split(":")
            pred = (int(start_token), int(end_token))
        elif pred in ["YES", "NO", ""]:
            pass
        else:
            raise InvalidSubmissionError(f"Invalid submission format: {pred}")

        # Parse valid answers
        if sample_id.endswith("_long"):
            sample_id = sample_id.split("_long")[0]
            annotation = true_annotations[sample_id]
            valid_answers = []
            if annotation["long_answer"]["start_token"] != -1:
                valid_answers.append(
                    (
                        int(annotation["long_answer"]["start_token"]),
                        int(annotation["long_answer"]["end_token"]),
                    ),
                )
        elif sample_id.endswith("_short"):
            sample_id = sample_id.split("_short")[0]
            annotation = true_annotations[sample_id]
            valid_answers = [
                (int(short_answer["start_token"]), int(short_answer["end_token"]))
                for short_answer in annotation["short_answers"]
            ]
            if annotation["yes_no_answer"] != "NONE":
                valid_answers.append(annotation["yes_no_answer"])
        else:
            raise InvalidSubmissionError(
                f"sample_id {sample_id} does not end with '_long' or '_short'"
            )

        # Mark as 0 or 1 based on the above rules
        if pred:
            if pred in valid_answers:
                y_true.append(1)
            else:
                y_true.append(0)
        else:
            if valid_answers:
                y_true.append(1)
            else:
                y_true.append(0)

    assert len(y_true) == len(y_pred), f"y_true {len(y_true)} != y_pred {len(y_pred)}"
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: dict) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")
