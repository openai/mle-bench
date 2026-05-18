import shutil
from pathlib import Path

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create test and test without labels
    test = read_csv(raw / "test_private_expanded.csv")
    test_without_labels = test[["id", "comment_text"]]

    # Copy over files
    shutil.copy(raw / "test_private_expanded.csv", private / "test.csv")
    shutil.copy(raw / "train.csv", public / "train.csv")
    test_without_labels.to_csv(public / "test.csv", index=False)

    # Create a sample submission file
    submission_df = test_without_labels.copy()
    submission_df = submission_df.drop(columns=["comment_text"])
    submission_df["prediction"] = 0.0
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Checks
    assert test.shape == (
        len(test),
        45,
    ), "test.csv should have 45 columns as per raw data"
    assert test_without_labels.shape == (
        len(test_without_labels),
        2,
    ), "test.csv should have 2 columns: id, comment_text"

    assert submission_df.shape == (
        len(test),
        2,
    ), "sample_submission.csv should have 2 columns: id, prediction"
