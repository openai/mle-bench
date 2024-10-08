import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError

from .kuzushiji_metric import kuzushiji_f1


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    No changes needed, since kuzushiji_metric.kuzushiji_f1 expects the submission and answers
    exactly as they are provided. In this function we just check for invalid submissions.
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows, but expected {len(answers)} rows."
        )
    if not {"image_id", "labels"}.issubset(submission.columns):
        raise InvalidSubmissionError("Submission must have columns 'image_id' and 'labels'.")

    submission = submission.sort_values("image_id").reset_index(drop=True)
    answers = answers.sort_values("image_id").reset_index(drop=True)

    if not all(submission["image_id"] == answers["image_id"]):
        raise InvalidSubmissionError("Submission image_id codes don't match solution.")

    # Check that labels have the correct format
    for pred in submission["labels"]:
        # Skip empty predictions
        if pd.isna(pred):
            continue

        # Pred is a string of the form "label1 X1 Y1 label2 X2 Y2 ..."
        # where label must be a str and X, Y are floats
        preds = pred.split(" ")
        if len(preds) % 3 != 0:
            raise InvalidSubmissionError(f"Malformed prediction string: {pred}")
        # Check that the X and Y values can be converted to floats
        if not all(isinstance(float(preds[i]), float) for i in range(1, len(preds), 3)) or not all(
            isinstance(float(preds[i]), float) for i in range(2, len(preds), 3)
        ):
            raise InvalidSubmissionError(f"X and Y values must be floats: {pred}")

    return submission, answers


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Submissions are evaluated on a modified version of the F1 Score. To score a true positive,
    you must provide center point coordinates that are within the ground truth bounding box and
    a matching label. The ground truth bounding boxes are defined in the format {label X Y Width Height},
    so if the ground truth label is U+003F 1 1 10 10 then a prediction of U+003F 3 3 would pass.

    See https://www.kaggle.com/competitions/kuzushiji-recognition/overview for more details.
    """
    submission, answers = prepare_for_metric(submission, answers)
    return float(kuzushiji_f1(sub=submission, solution=answers))
