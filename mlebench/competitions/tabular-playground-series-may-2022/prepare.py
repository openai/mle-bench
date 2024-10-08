from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    old_train = read_csv(raw / "train.csv")

    # 900k train, 1.6m - 900k = 700k test; so 700k/1.6m = 0.4375
    # We create our split at 100,000 test samples to get same OOM while keeping as many samples as possible in train
    new_train, new_test = train_test_split(old_train, test_size=100_000, random_state=0)

    # make ids go from 0 to len(new_train) - 1
    new_train.id = np.arange(len(new_train))
    # and from len(new_train) to len(new_train) + len(new_test) - 1
    new_test.id = np.arange(len(new_train), len(new_train) + len(new_test))

    # make downstream files
    new_test_without_labels = new_test.drop(columns=["target"]).copy()
    gold_submission = new_test[["id", "target"]].copy()
    sample_submission = gold_submission.copy()
    sample_submission.target = 0.5

    # save
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    gold_submission.to_csv(private / "gold_submission.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Expected the sum of the lengths of the new train and test to be equal to the length of the original train."
    assert len(new_test) == len(
        sample_submission
    ), "Expected the length of the private test to be equal to the length of the sample submission."
    assert len(new_test) == len(
        gold_submission
    ), "Expected the length of the private test to be equal to the length of the gold submission."

    assert (
        new_train.columns.to_list() == old_train.columns.to_list()
    ), "Expected the columns of the new train to be the same as the columns of the original train."
    assert (
        new_test.columns.to_list() == old_train.columns.to_list()
    ), "Expected the columns of the new test to be the same as the columns of the original train"

    # check that ids dont overlap between train and test
    assert set(new_train.id).isdisjoint(
        set(new_test.id)
    ), "Expected the ids of the new train and test to be disjoint."
