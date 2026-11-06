import math

import pandas as pd
import pytest

from mlebench.grade_helpers import Grader, InvalidSubmissionError


@pytest.fixture(
    params=[
        pytest.param(
            (
                "champs-scalar-coupling",
                "id",
                "scalar_coupling_constant",
                (math.log(2) + math.log(5)) / 2,
            ),
            id="champs",
        ),
        pytest.param(
            ("predict-volcanic-eruptions-ingv-oe", "segment_id", "time_to_eruption", 4),
            id="volcanic",
        ),
    ]
)
def grading_case(request):
    competition_id, id_column, target_column, expected_score = request.param
    answers = pd.DataFrame({id_column: [1, 2, 3, 4], target_column: [10, 20, 30, 40]})
    if competition_id == "champs-scalar-coupling":
        answers["type"] = ["1JHC", "1JHC", "2JHC", "2JHC"]

    submission = pd.DataFrame({id_column: [1, 2, 3, 4], target_column: [10, 24, 32, 48]})
    grader = Grader(
        name=competition_id,
        grade_fn=f"mlebench.competitions.{competition_id}.grade:grade",
    )
    return grader, submission, answers, id_column, target_column, expected_score


def test_unique_submission_ids_preserve_shuffled_scores(grading_case):
    grader, submission, answers, _, _, expected_score = grading_case
    shuffled = submission.iloc[[2, 0, 3, 1]]

    # The absolute errors are 0, 4, 2, 8. Volcanic rounds their mean, 3.5,
    # to 4; CHAMPS averages log(2) and log(5) for its two coupling types.
    assert grader.grade_fn(shuffled, answers) == pytest.approx(expected_score)
    assert grader(shuffled, answers) == pytest.approx(round(expected_score, 5))


@pytest.mark.parametrize("prediction_offset", [0, 100], ids=["identical", "conflicting"])
def test_duplicate_submission_ids_are_invalid(grading_case, prediction_offset):
    grader, submission, answers, _, target_column, _ = grading_case
    duplicate = submission.iloc[[0]].copy()
    duplicate[target_column] += prediction_offset
    submission = pd.concat([submission, duplicate, duplicate], ignore_index=True)

    with pytest.raises(InvalidSubmissionError, match="not unique"):
        grader.grade_fn(submission, answers)

    assert grader(submission, answers) is None


@pytest.mark.parametrize("unknown_id", [False, True], ids=["missing", "unexpected"])
def test_submission_id_set_must_still_match_answers(grading_case, unknown_id):
    grader, submission, answers, id_column, _, _ = grading_case
    if unknown_id:
        submission = submission.copy()
        submission.loc[0, id_column] = 99
    else:
        submission = submission.iloc[1:]

    with pytest.raises(InvalidSubmissionError):
        grader.grade_fn(submission, answers)

    assert grader(submission, answers) is None
