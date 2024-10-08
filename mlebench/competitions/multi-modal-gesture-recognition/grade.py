from typing import List

import numpy as np
import pandas as pd
from Levenshtein import distance

from mlebench.grade_helpers import InvalidSubmissionError


def final_edit_distance(y_true: List[int], y_pred: List[int]) -> float:
    """
    > The overall score we compute is the sum of the Levenshtein distances for all the lines of the
      result file compared to the corresponding lines in the truth value file, divided by the total
      number of gestures in the truth value file. This score is analogous to an error rate.
      However, it can exceed one.
      (https://www.kaggle.com/competitions/multi-modal-gesture-recognition/overview)
    """
    sum_of_distances = np.sum([distance(a, b) for a, b in zip(y_true, y_pred)])
    total_num_gestures = sum(len(x) for x in y_true)
    return sum_of_distances / total_num_gestures


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame):
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    if "Id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an 'Id' column")

    if "Sequence" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'Sequence' column")

    assert "Id" in answers.columns, "Answers must have 'Id' column"
    assert "Sequence" in answers.columns, "Answers must have 'Sequence' column"

    submission = submission.sort_values("Id")
    answers = answers.sort_values("Id")

    if (submission["Id"].values != answers["Id"].values).any():
        raise InvalidSubmissionError("Submission and answers must have the same ids")

    # Read as strings, convert to list of numbers
    submission["Sequence"] = submission["Sequence"].astype(str)
    answers["Sequence"] = answers["Sequence"].astype(str)
    answers_sequences = [list(map(int, x.split())) for x in answers["Sequence"]]
    try:
        submission_sequences = [list(map(int, x.split())) for x in submission["Sequence"]]
    except ValueError as e:
        raise InvalidSubmissionError(
            f"Submission sequences must be integers separated by spaces. Failed to convert Sequence to list of integers: {e}"
        )

    return answers_sequences, submission_sequences


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return final_edit_distance(y_true, y_pred)
