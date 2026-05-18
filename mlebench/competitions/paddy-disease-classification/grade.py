from pandas import DataFrame
from sklearn.metrics import accuracy_score

from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: DataFrame, answers: DataFrame) -> float:
    expected_columns = set(["image_id", "label"])

    assert set(answers.columns).issuperset(
        expected_columns
    ), f"Expected columns {expected_columns}, but got {set(answers.columns)}"

    assert not answers.isnull().values.any(), "Answers contain NaN values"

    if not set(submission.columns).issuperset(expected_columns):
        raise InvalidSubmissionError(
            f"Expected columns {expected_columns}, but got {set(submission.columns)}"
        )

    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Expected {len(answers)} rows, but got {len(submission)}")

    sorted_submission = submission.sort_values(by="image_id").reset_index(drop=True)
    sorted_answers = answers.sort_values(by="image_id").reset_index(drop=True)

    if (sorted_submission["image_id"].values != sorted_answers["image_id"].values).any():
        raise InvalidSubmissionError("Invalid image IDs in the submission!")

    y_pred = sorted_submission["label"].fillna("").tolist()
    y_true = sorted_answers["label"]

    score = accuracy_score(y_true=y_true, y_pred=y_pred)

    return score
