import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    old_train = pd.read_csv(raw / "train_labels.csv")

    num_test = len(list((raw / "test").glob("*.tif")))
    test_ratio = num_test / (len(old_train) + num_test)

    new_train_ids, new_test_ids = train_test_split(
        old_train["id"], test_size=test_ratio, random_state=0
    )
    new_train = old_train[old_train["id"].isin(new_train_ids)]
    new_test = old_train[old_train["id"].isin(new_test_ids)]

    assert set(new_train["id"]).isdisjoint(
        set(new_test["id"])
    ), "Train should not contain id's of test images"
    assert len(new_train_ids) + len(new_test_ids) == len(
        old_train
    ), "The combined length of new_train_ids and new_test_ids should equal the length of old_train"

    # Copy over files
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in tqdm(new_train_ids, desc="Copying train images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.tif",
            dst=public / "train" / f"{file_id}.tif",
        )
    for file_id in tqdm(new_test_ids, desc="Copying test images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.tif",
            dst=public / "test" / f"{file_id}.tif",
        )

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["label"] = 0

    # Copy over files
    new_train.to_csv(public / "train_labels.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Check integrity of files copied
    assert len(list(public.glob("train/*.tif"))) == len(
        new_train_ids
    ), "Number of train images should be equal to the number of unique id's in the train set"
    assert len(list(public.glob("test/*.tif"))) == len(
        new_test_ids
    ), "Number of test images should be equal to the number of unique id's in the test set"
