import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    old_train = read_csv(raw / "train.csv")
    # drop image ce27d36c9147cc19.jpg: it appears twice and may occur across train and test when split
    old_train = old_train[old_train["image"] != "ce27d36c9147cc19.jpg"]

    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    sample_submission = new_test.copy()[["image", "hotel_id"]]
    sample_submission["hotel_id"] = "36363 53586 18807 64314 60181"

    # save public files
    new_train.to_csv(public / "train.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # copy images from raw to prepared/public accordingly
    (public / "train_images").mkdir(exist_ok=True, parents=True)
    for image, chain in tqdm(
        zip(new_train["image"], new_train["chain"]), total=len(new_train), desc="Train images"
    ):
        chain = str(chain)
        (public / "train_images" / chain).mkdir(exist_ok=True, parents=True)
        shutil.copy(raw / "train_images" / chain / image, public / "train_images" / chain / image)

    (public / "test_images").mkdir(exist_ok=True, parents=True)
    for image, chain in tqdm(
        zip(new_test["image"], new_test["chain"]), total=len(new_test), desc="Test images"
    ):
        chain = str(chain)
        (public / "test_images").mkdir(exist_ok=True, parents=True)
        shutil.copy(raw / "train_images" / chain / image, public / "test_images" / image)

    # save private files
    new_test.to_csv(private / "test.csv", index=False)

    # checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
    assert sample_submission.columns.to_list() == [
        "image",
        "hotel_id",
    ], "Sample submission columns should only be `image` and `hotel_id`"
    assert len(sample_submission) == len(new_test), "Sample submission length should match test set"
    for image, chain in zip(new_train["image"], new_train["chain"]):
        chain = str(chain)
        assert (
            public / "train_images" / chain / image
        ).exists(), f"Image {image} not found in train_images folder"
    for image in new_test["image"]:
        assert (
            public / "test_images" / image
        ).exists(), f"Image {image} not found in test_images folder"
    assert not set(new_train["image"]).intersection(
        set(new_test["image"])
    ), "Train and test ids overlap"
