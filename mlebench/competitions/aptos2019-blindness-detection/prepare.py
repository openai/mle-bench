import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["diagnosis"])

    (public / "test_images").mkdir(exist_ok=True)
    (public / "train_images").mkdir(exist_ok=True)

    # Copy data
    for file_id in new_train["id_code"]:
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}.png",
            dst=public / "train_images" / f"{file_id}.png",
        )

    for file_id in new_test_without_labels["id_code"]:
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}.png",
            dst=public / "test_images" / f"{file_id}.png",
        )

    # Check integrity of the files copied
    assert set(new_train["id_code"]).isdisjoint(
        set(new_test["id_code"])
    ), "Train and test sets should have no shared ids"

    assert len(new_test_without_labels) == len(
        new_test
    ), "Public and Private tests should have equal length"

    assert len(list(public.glob("train_images/*.png"))) == len(
        new_train
    ), "Public train images should have the same number of images as the length of train set"

    assert len(list(public.glob("test_images/*.png"))) == len(
        new_test_without_labels
    ), "Public test images should have the same number of images as the length of test set"

    train_image_files = set(public.glob("train_images/*.png"))
    test_image_files = set(public.glob("test_images/*.png"))
    common_files = train_image_files.intersection(test_image_files)
    assert not common_files, f"Images found in both train_images and test_images: {common_files}"

    for file_id in new_test["id_code"]:
        assert (
            public / "test_images" / f"{file_id}.png"
        ).exists(), f"Image file for {file_id} not found in test_images"

    for file_id in new_train["id_code"]:
        assert (
            public / "train_images" / f"{file_id}.png"
        ).exists(), f"Image file for {file_id} not found in train_images"

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df["diagnosis"] = 0

    # Write CSVs
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
