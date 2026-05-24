import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import mlebench.grade as grade_module
from mlebench.grade import validate_submission
from mlebench.grade_helpers import Grader
from mlebench.registry import Competition


def _grade_expecting_jsonl_answers(submission: pd.DataFrame, answers: list) -> float:
    """Grader that expects `answers` to be a list of dicts, as loaded from a JSONL file.

    Mirrors how graders for competitions with `.jsonl` answer files (e.g.
    tensorflow2-question-answering) consume their answers.
    """
    return float(len([row["example_id"] for row in answers]))


def _make_competition(answers_path: Path, submission_path: Path, tmp: Path) -> Competition:
    grader = Grader(
        name="jsonl-answers-grader",
        grade_fn=f"{__name__}:_grade_expecting_jsonl_answers",
    )
    return Competition(
        id="jsonl-answers-comp",
        name="JSONL answers competition",
        description="Fixture competition whose answers are stored as JSONL.",
        grader=grader,
        answers=answers_path,
        gold_submission=answers_path,
        sample_submission=submission_path,
        competition_type="test",
        prepare_fn=lambda raw, public, private: raw,
        raw_dir=tmp,
        private_dir=tmp,
        public_dir=tmp,
        checksums=tmp / "checksums.yaml",
        leaderboard=tmp / "leaderboard.csv",
    )


def test_validate_submission_handles_jsonl_answers(monkeypatch):
    """validate_submission must load answers via load_answers so JSONL answer
    files are parsed correctly, matching grade_csv. Regression test for the bug
    where read_csv was used on `.jsonl` answers, failing every valid submission."""
    monkeypatch.setattr(
        grade_module, "is_dataset_prepared", lambda competition, grading_only=False: True
    )

    with TemporaryDirectory() as d:
        tmp = Path(d)
        answers_path = tmp / "test.jsonl"
        with answers_path.open("w") as f:
            f.write(json.dumps({"example_id": "a"}) + "\n")
            f.write(json.dumps({"example_id": "b"}) + "\n")

        submission_path = tmp / "submission.csv"
        submission_path.write_text("example_id,PredictionString\na,\nb,\n")

        competition = _make_competition(answers_path, submission_path, tmp)

        is_valid, message = validate_submission(submission_path, competition)

    assert is_valid, message
    assert message == "Submission is valid."
