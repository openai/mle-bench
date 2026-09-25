import importlib

import numpy as np
import pandas as pd
import pytest

from mlebench.grade_helpers import InvalidSubmissionError

grade_module = importlib.import_module(
    "mlebench.competitions.osic-pulmonary-fibrosis-progression.grade"
)
grade = grade_module.grade

PATIENTS = ["A", "B", "C", "D"]
SCORED_WEEKS = [30, 40, 50]
UNSCORED_WEEK = 10  # rows with a NaN FVC in the answers are dummy rows that are not graded
CONFIDENCE = 100.0


def _answers() -> pd.DataFrame:
    rows = []
    for patient in PATIENTS:
        rows.append(
            {"Patient_Week": f"{patient}_{UNSCORED_WEEK}", "FVC": np.nan, "Patient": patient}
        )
        for week in SCORED_WEEKS:
            rows.append({"Patient_Week": f"{patient}_{week}", "FVC": 2000.0, "Patient": patient})
    return pd.DataFrame(rows)


def _submission(error_by_patient: dict) -> pd.DataFrame:
    answers = _answers()
    submission = answers[["Patient_Week", "FVC"]].copy()
    submission["FVC"] = 2000.0 + answers["Patient"].map(error_by_patient).fillna(0.0)
    submission["Confidence"] = CONFIDENCE
    return submission


PERFECT = -np.log(np.sqrt(2) * CONFIDENCE)
OFF_BY_500 = -np.sqrt(2) * 500 / CONFIDENCE - np.log(np.sqrt(2) * CONFIDENCE)


def test_full_submission_is_scored_over_every_scored_row():
    submission = _submission({"C": 500.0, "D": 500.0})
    assert grade(submission, _answers()) == pytest.approx((PERFECT + OFF_BY_500) / 2)


def test_full_perfect_submission_score():
    assert grade(_submission({}), _answers()) == pytest.approx(PERFECT)


def test_submission_without_the_poorly_predicted_patients_is_rejected():
    submission = _submission({"C": 500.0, "D": 500.0})
    submission = submission[submission["Patient_Week"].str.startswith(("A_", "B_"))]
    with pytest.raises(InvalidSubmissionError, match="missing"):
        grade(submission, _answers())


def test_submission_missing_one_scored_row_is_rejected():
    submission = _submission({})
    submission = submission[submission["Patient_Week"] != "B_40"]
    with pytest.raises(InvalidSubmissionError, match="B_40"):
        grade(submission, _answers())


def test_submission_with_only_unscored_rows_is_rejected():
    submission = _submission({})
    submission = submission[submission["Patient_Week"].str.endswith(f"_{UNSCORED_WEEK}")]
    with pytest.raises(InvalidSubmissionError, match="missing"):
        grade(submission, _answers())


def test_row_not_in_answers_is_still_rejected():
    submission = _submission({})
    extra = pd.DataFrame([{"Patient_Week": "Z_99", "FVC": 2000.0, "Confidence": CONFIDENCE}])
    with pytest.raises(InvalidSubmissionError, match="does not exist"):
        grade(pd.concat([submission, extra], ignore_index=True), _answers())
