import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")

    old_train["split"] = "undecided"
    target_test_size = 0.1

    # seeded random generator for numpy
    np_rng = np.random.default_rng(0)

    # ensure each id occurs in train and test set at least once
    # when there's only one image for an id, goes randomly to train or test
    whale_ids = old_train["Id"].unique()
    for whale_id in whale_ids:
        whale_images = old_train[old_train["Id"] == whale_id]
        if len(whale_images) >= 2:
            # randomly assign one of these to train and one to test
            selected = whale_images.sample(2, random_state=0)
            old_train.loc[selected.index[0], "split"] = "train"
            old_train.loc[selected.index[1], "split"] = "test"
        else:
            # randomly assign this one image to train or test
            old_train.loc[whale_images.index[0], "split"] = np_rng.choice(
                ["train", "test"], replace=False, p=[1 - target_test_size, target_test_size]
            )

    # split the remaining data
    remaining_data = old_train[old_train["split"] == "undecided"]
    train, test = train_test_split(remaining_data, test_size=target_test_size, random_state=0)
    old_train.loc[train.index, "split"] = "train"
    old_train.loc[test.index, "split"] = "test"

    # finally, can split out into separate dataframes
    new_train = old_train[old_train["split"] == "train"].drop(columns=["split"]).copy()
    answers = old_train[old_train["split"] == "test"].drop(columns=["split"]).copy()

    # If a whale Id is only in the test set, it should be labeled as new_whale instead
    ids_in_test_but_not_train = set(answers["Id"]) - set(new_train["Id"])
    answers.loc[answers["Id"].isin(ids_in_test_but_not_train), "Id"] = "new_whale"

    # Create sample submission
    sample_submission = answers.copy()
    sample_submission["Id"] = "new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46"

    # Checks
    assert len(answers) == len(
        sample_submission
    ), "Answers and sample submission should have the same length"
    assert new_train.shape[1] == 2, "Train should have exactly 2 columns"
    assert sample_submission.shape[1] == 2, "Sample submission should have exactly 2 columns"
    assert answers.shape[1] == 2, "Answers should have exactly 2 columns"
    assert (
        "new_whale" in answers["Id"].values
    ), "Answers should contain at least some values with 'new_whale' in the 'Id' column"
    assert len(new_train) + len(answers) == len(
        old_train
    ), "The combined length of new_train and answers should equal the length of old_train"

    # Write CSVs
    answers.to_csv(private / "test.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Copy over files
    (public / "test").mkdir(exist_ok=True)
    (public / "train").mkdir(exist_ok=True)

    for file_id in tqdm(new_train["Image"], desc="Copying train images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}",
            dst=public / "train" / f"{file_id}",
        )

    for file_id in tqdm(answers["Image"], desc="Copying test images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}",
            dst=public / "test" / f"{file_id}",
        )

    # File checks
    train_files = list(public.glob("train/*.jpg"))
    test_files = list(public.glob("test/*.jpg"))
    assert len(train_files) == len(
        new_train
    ), "Train dir should have the same number of images as the length of train set"
    assert len(test_files) == len(
        answers
    ), "Test dir should have the same number of images as the length of test set"
    assert not set(train_files) & set(test_files), "Train and test files should be distinct"
