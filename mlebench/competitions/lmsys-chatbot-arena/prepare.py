import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import extract, read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    # Original train has 55k rows. Original hidden test has 25k rows. We make a new test set with 10% of the original train.
    new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test = answers[["id", "prompt", "response_a", "response_b"]].copy()

    sample_submission = answers[["id"]].copy()
    sample_submission["winner_model_a"] = 0.3333333333333333
    sample_submission["winner_model_b"] = 0.3333333333333333
    sample_submission["winner_tie"] = 0.3333333333333333

    # Checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"New train and test should have the same number of rows as the original train"
    assert set(new_train["id"]).isdisjoint(
        set(new_test["id"])
    ), f"New train and test should have no overlapping ids"
    assert new_test.columns.tolist() == [
        "id",
        "prompt",
        "response_a",
        "response_b",
    ], f"New test should have columns id, prompt, response_a, response_b"
    assert sample_submission.columns.tolist() == [
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ], f"Sample submission should have columns id, winner_model_a, winner_model_b, winner_tie"
    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), f"New train should have the same columns as the original train"

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
