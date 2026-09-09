from importlib import import_module
from io import StringIO

import pandas as pd
import pytest

from mlebench.grade_helpers import Grader, InvalidSubmissionError


@pytest.fixture(params=["english", "russian"])
def grader_module(request):
    return import_module(
        f"mlebench.competitions.text-normalization-challenge-{request.param}-language.grade"
    )


def test_missing_tokens_preserve_string_accuracy(grader_module):
    answers = pd.read_csv(StringIO("id,after\n0_0,hello\n0_1,\n0_2,NA\n0_3,123\n"))
    submission = answers.iloc[::-1].copy()

    assert grader_module.grade(submission, answers) == 1.0
    submission.loc[submission["id"] == "0_0", "after"] = "different"
    assert grader_module.grade(submission, answers) == 0.75
    grader = Grader("accuracy", f"{grader_module.__name__}:grade")
    assert grader(submission, answers) == 0.75


def test_mismatched_ids_are_still_rejected(grader_module):
    answers = pd.DataFrame({"id": ["0_0"], "after": ["hello"]})
    submission = pd.DataFrame({"id": ["0_1"], "after": ["hello"]})
    with pytest.raises(InvalidSubmissionError, match="do not match"):
        grader_module.grade(submission, answers)
