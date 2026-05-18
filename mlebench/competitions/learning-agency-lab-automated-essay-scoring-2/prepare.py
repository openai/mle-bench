from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from original train split
    old_train = read_csv(raw / "train.csv")
    # Original train has 17307 rows. Original hidden test has approx 8k rows. We just take 10% of the original train as the test set.
    new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test = answers.drop(columns=["score"])

    sample_submission = answers[["essay_id"]].copy()
    sample_submission["score"] = np.random.RandomState(42).randint(
        1, 7, size=len(sample_submission)
    )

    # Checks
    assert set(new_train["essay_id"]).isdisjoint(
        set(new_test["essay_id"])
    ), "Essay IDs in train and test sets are not disjoint"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Train and test sets do not sum to original train set"
    assert len(new_test) == len(
        sample_submission
    ), f"Test and sample submission sets do not have the same length"
    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), f"Train set columns do not match original train set, got {new_train.columns.tolist()}"
    assert new_test.columns.tolist() == [
        "essay_id",
        "full_text",
    ], f"Test set columns do not match expected columns, got {new_test.columns.tolist()}"
    assert sample_submission.columns.tolist() == [
        "essay_id",
        "score",
    ], f"Sample submission set columns do not match expected columns, got {sample_submission.columns.tolist()}"

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
