import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")

    # Train is c. 55M rows, original test is 9914 rows
    new_train, new_test = train_test_split(old_train, test_size=9914, random_state=0)
    new_test_without_labels = new_test.drop(columns=["fare_amount"])

    # Create a sample submission file
    submission_df = new_test.copy()[["key"]]
    submission_df["fare_amount"] = 11.35

    # Write CSVs
    new_train.to_csv(public / "labels.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # Copy over other files
    shutil.copy(raw / "GCP-Coupons-Instructions.rtf", public / "GCP-Coupons-Instructions.rtf")

    # Checks
    assert set(new_train["key"]).isdisjoint(
        set(new_test["key"])
    ), "Train and test sets share samples!"
    assert new_test.shape[1] == 8, f"Test set should have 8 columns, but has {new_test.shape[1]}"
    assert (
        new_test_without_labels.shape[1] == 7
    ), f"Test set without labels should have 7 columns, but has {new_test_without_labels.shape[1]}"
    assert new_train.shape[1] == 8, f"Train set should have 8 columns, but has {new_train.shape[1]}"
    assert (
        submission_df.shape[1] == 2
    ), f"Sample submission should have 2 columns, but has {submission_df.shape[1]}"
    assert (
        submission_df.shape[0] == new_test.shape[0]
    ), f"Sample submission should have {new_test.shape[0]} rows, but has {submission_df.shape[0]}"
