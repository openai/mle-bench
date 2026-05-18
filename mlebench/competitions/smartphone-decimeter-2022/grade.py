import numpy as np
from pandas import DataFrame

from mlebench.grade_helpers import InvalidSubmissionError

from .notebook import calc_score as get_score


def grade(submission: DataFrame, answers: DataFrame) -> float:
    assert "tripId" in answers.columns, "Answers DataFrame must have a 'tripId' column."
    assert (
        "UnixTimeMillis" in answers.columns
    ), "Answers DataFrame must have a 'UnixTimeMillis' column."
    assert (
        "LatitudeDegrees" in answers.columns
    ), "Answers DataFrame must have a 'LatitudeDegrees' column."
    assert (
        "LongitudeDegrees" in answers.columns
    ), "Answers DataFrame must have a 'LongitudeDegrees' column."

    if "tripId" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'tripId' column.")

    if "UnixTimeMillis" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'UnixTimeMillis' column.")

    if "LatitudeDegrees" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'LatitudeDegrees' column.")

    if "LongitudeDegrees" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have a 'LongitudeDegrees' column.")

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )

    scores = []
    trip_ids = answers["tripId"].unique()

    for trip_id in trip_ids:
        filtered_answers = answers[answers["tripId"] == trip_id]
        filtered_submission = submission[submission["tripId"] == trip_id]

        if len(filtered_submission) != len(filtered_answers):
            raise InvalidSubmissionError(
                "Submission and answers DataFrames must have the same number of rows for each 'tripId'."
                f"Got {len(filtered_submission)} rows in the submission and {len(filtered_answers)} rows "
                f"in the answers for tripId {trip_id}."
            )

        score = get_score(filtered_submission, filtered_answers)
        scores.append(score)

    return np.mean(scores)
