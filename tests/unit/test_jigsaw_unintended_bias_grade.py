from importlib import import_module

import pandas as pd
import pytest

from mlebench.grade_helpers import Grader

COMPETITION = "jigsaw-unintended-bias-in-toxicity-classification"
GRADE_PATH = f"mlebench.competitions.{COMPETITION}.grade:grade"
grade = import_module(f"mlebench.competitions.{COMPETITION}.grade").grade
IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]


@pytest.fixture
def answers():
    # Both target classes occur inside and outside every identity subgroup.
    return pd.DataFrame(
        {
            "id": range(8),
            "toxicity": [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6],
            **{column: [0.25] * 4 + [0.75] * 4 for column in IDENTITY_COLUMNS},
        }
    )


@pytest.mark.parametrize("prediction_column", ["prediction", "toxicity"])
@pytest.mark.parametrize(
    "predictions, expected_score",
    [
        ([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6], 1.0),
        ([0.1, 0.6, 0.8, 0.9, 0.2, 0.3, 0.4, 0.7], 0.75),
    ],
)
def test_auc_preserves_prediction_rankings(answers, prediction_column, predictions, expected_score):
    grader = Grader(COMPETITION, GRADE_PATH)
    for scale in (1.0, 0.49):
        submission = pd.DataFrame(
            {"id": answers["id"], prediction_column: pd.Series(predictions) * scale}
        ).iloc[::-1]

        assert grader(submission, answers.copy()) == pytest.approx(expected_score)
        assert grade(submission, answers.copy()) == pytest.approx(expected_score)


@pytest.mark.parametrize("prediction_column", ["prediction", "toxicity"])
def test_nan_predictions_are_not_scored(answers, prediction_column):
    submission = answers[["id", "toxicity"]].rename(columns={"toxicity": prediction_column})
    submission.loc[0, prediction_column] = float("nan")

    assert Grader(COMPETITION, GRADE_PATH)(submission, answers.copy()) is None
    with pytest.raises(ValueError, match="NaN"):
        grade(submission, answers.copy())
