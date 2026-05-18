from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["Cover_Type"])

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Create a sample submission file
    submission_df = pd.DataFrame(
        {
            "Id": new_test["Id"].values,
            "Cover_Type": 2,
        }
    )
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    assert len(new_train) + len(new_test) == len(
        old_train
    ), "train and test splits lengths do not add up to original data."
    assert set(new_train["Id"]).isdisjoint(
        set(new_test["Id"])
    ), f"there are Ids in both train and test sets: {set(new_train['Id']).intersection(set(new_test['Id']))}"
    assert len(new_test_without_labels) == len(
        new_test
    ), "public and Private tests should have equal length"
