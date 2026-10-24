"""Unit tests for submission validation and grading."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlebench.grade import validate_submission


def test_validate_submission_with_jsonl_answers(tmp_path: Path):
    """
    Verify that validate_submission correctly uses load_answers,
    allowing competitions with JSONL answers (such as tensorflow2-question-answering)
    to parse answers as list[dict] rather than mis-parsing with read_csv.
    """
    submission_path = tmp_path / "submission.csv"
    submission_df = pd.DataFrame({
        "example_id": ["sample_1_long", "sample_1_short"],
        "PredictionString": ["1:10", "YES"],
    })
    submission_df.to_csv(submission_path, index=False)

    answers_path = tmp_path / "test.jsonl"
    sample_answer = {
        "example_id": "sample_1",
        "annotations": [],
        "document_text": "sample text",
    }
    answers_path.write_text(json.dumps(sample_answer) + "\n", encoding="utf-8")

    received_answers = []

    def mock_grade_fn(sub: pd.DataFrame, ans):
        received_answers.append(ans)
        # Verify ans is a list of dicts, as expected by JSONL answer competitions
        assert isinstance(ans, list), f"Expected list[dict], got {type(ans)}"
        assert isinstance(ans[0], dict), f"Expected dict, got {type(ans[0])}"
        return 1.0

    mock_competition = MagicMock()
    mock_competition.id = "test-jsonl-competition"
    mock_competition.answers = answers_path
    mock_competition.grader.grade_fn = mock_grade_fn

    with patch("mlebench.grade.is_dataset_prepared", return_value=True):
        is_valid, msg = validate_submission(submission_path, mock_competition)

    assert is_valid is True
    assert "Submission is valid" in msg
    assert len(received_answers) == 1
    assert received_answers[0] == [sample_answer]


def test_validate_submission_with_csv_answers(tmp_path: Path):
    """Verify that validate_submission also works seamlessly with standard CSV answers."""
    submission_path = tmp_path / "submission.csv"
    submission_df = pd.DataFrame({"id": [1, 2], "pred": [0.1, 0.9]})
    submission_df.to_csv(submission_path, index=False)

    answers_path = tmp_path / "test.csv"
    answers_df = pd.DataFrame({"id": [1, 2], "label": [0, 1]})
    answers_df.to_csv(answers_path, index=False)

    def mock_grade_fn(sub: pd.DataFrame, ans: pd.DataFrame):
        assert isinstance(ans, pd.DataFrame)
        assert list(ans["id"]) == [1, 2]
        return 0.5

    mock_competition = MagicMock()
    mock_competition.id = "test-csv-competition"
    mock_competition.answers = answers_path
    mock_competition.grader.grade_fn = mock_grade_fn

    with patch("mlebench.grade.is_dataset_prepared", return_value=True):
        is_valid, msg = validate_submission(submission_path, mock_competition)

    assert is_valid is True
    assert "Submission is valid" in msg
