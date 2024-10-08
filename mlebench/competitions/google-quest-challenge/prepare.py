from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):

    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # question_type_spelling is almost always 0; if entirely 0 in test set, swap one row
    if new_test["question_type_spelling"].nunique() == 1:
        # need to do this swapping because spearmanr needs variation in the data to work
        suitable_train_row_index = new_train[new_train["question_type_spelling"] != 0].index[0]
        suitable_test_row_index = 0
        temp = new_test.iloc[suitable_test_row_index].copy()
        new_test.iloc[suitable_test_row_index] = new_train.loc[suitable_train_row_index].copy()
        new_train.loc[suitable_train_row_index] = temp

    new_test_without_labels = new_test.drop(CLASSES, axis=1, inplace=False)

    # Create sample submission; private test will match this format
    cols_to_keep = ["qa_id"] + CLASSES
    new_test = new_test[cols_to_keep]
    sample_submission = new_test.copy()
    # spearmanr needs variation in the data to work; make each column increasing from 0 to 1
    n, M = len(sample_submission), len(CLASSES)
    sample_submission[CLASSES] = np.tile(np.linspace(0, 1, n)[:, None], (1, M))

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Checks
    assert new_test_without_labels.shape[1] == 11, "Public test set should have 11 columns"
    assert new_train.shape[1] == 41, "Public train set should have 41 columns"
    # each private test set target column should not be constant
    for column in CLASSES:
        assert (
            new_test[column].nunique() > 1
        ), f"Column {column} should not be constant in the private test set"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
    assert (
        sample_submission.columns.to_list() == new_test.columns.to_list()
    ), "Sample submission columns should match test set"
    assert len(sample_submission) == len(new_test), "Sample submission length should match test set"
