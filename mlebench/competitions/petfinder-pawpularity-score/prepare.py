import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    old_train = read_csv(raw / "train.csv")

    np_rng = np.random.default_rng(0)

    # Original ratio: 6800/(9912 + 6800) = ~ 0.41 test_size
    # We use 0.1 ratio to avoid taking out too many samples from train
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    new_test_without_labels = new_test.drop(columns=["Pawpularity"])

    # random floats between 1 and 100 inclusive, with 2 decimal places
    sample_submission = new_test[["Id", "Pawpularity"]].copy()
    sample_submission["Pawpularity"] = np_rng.uniform(1, 100, len(sample_submission)).round(2)

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    (public / "train").mkdir(exist_ok=True)
    for img_id in tqdm(new_train["Id"], desc="Copying train images", total=len(new_train)):
        shutil.copy(raw / "train" / f"{img_id}.jpg", public / "train" / f"{img_id}.jpg")

    (public / "test").mkdir(exist_ok=True)
    for img_id in tqdm(
        new_test_without_labels["Id"],
        desc="Copying test images",
        total=len(new_test_without_labels),
    ):
        shutil.copy(raw / "train" / f"{img_id}.jpg", public / "test" / f"{img_id}.jpg")

    # checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Train and test length should sum to the original train length"
    assert len(sample_submission) == len(
        new_test
    ), "Sample submission should have the same length as the test set"

    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), "Old and new train columns should match"
    assert (
        new_test_without_labels.columns.tolist() == new_train.columns.tolist()[:-1]
    ), "Public test columns should match train columns, minus the target column"
    assert (
        new_test.columns.tolist() == new_train.columns.tolist()
    ), "Private test columns should match train columns"
    assert sample_submission.columns.tolist() == [
        "Id",
        "Pawpularity",
    ], "Sample submission columns should be Id, Pawpularity"

    assert set(new_train["Id"]).isdisjoint(
        set(new_test["Id"])
    ), "Train and test ids should not overlap"

    # check copy was successful
    assert len(list((public / "train").glob("*.jpg"))) == len(
        new_train
    ), "Train images should match the train set"
    assert len(list((public / "test").glob("*.jpg"))) == len(
        new_test
    ), "Test images should match the test set"
