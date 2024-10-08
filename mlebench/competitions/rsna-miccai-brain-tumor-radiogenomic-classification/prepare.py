import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train and test splits from train set
    old_train = read_csv(raw / "train_labels.csv", dtype={"BraTS21ID": str, "MGMT_value": int})
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Copy over images
    (public / "train").mkdir(exist_ok=True)
    for file_id in new_train["BraTS21ID"]:
        (public / "train" / file_id).mkdir(exist_ok=True)
        shutil.copytree(
            src=raw / "train" / file_id,
            dst=public / "train" / file_id,
            dirs_exist_ok=True,
        )
    assert len(list(public.glob("train/*"))) == len(
        new_train
    ), "Public train should have the same number of images as the train set"

    (public / "test").mkdir(exist_ok=True)
    for file_id in new_test["BraTS21ID"]:
        (public / "test" / file_id).mkdir(exist_ok=True)
        shutil.copytree(
            src=raw / "train" / file_id,
            dst=public / "test" / file_id,
            dirs_exist_ok=True,
        )
    assert len(list(public.glob("test/*"))) == len(
        new_test
    ), "Public train should have the same number of images as the train set"

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df["MGMT_value"] = 0.5

    # Copy over files
    new_train.to_csv(public / "train_labels.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
