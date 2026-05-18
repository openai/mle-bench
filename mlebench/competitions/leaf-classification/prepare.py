import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import extract, read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # extract only what we need
    extract(raw / "train.csv.zip", raw)
    extract(raw / "images.zip", raw)

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["species"])

    # match the format of the sample submission
    new_test = new_test[["id", "species"]]
    new_test = df_to_one_hot(new_test, "id", "species", classes=CLASSES)

    (public / "images").mkdir(exist_ok=True)
    (private / "images").mkdir(exist_ok=True)

    for file_id in new_train["id"]:
        shutil.copyfile(
            src=raw / "images" / f"{file_id}.jpg",
            dst=public / "images" / f"{file_id}.jpg",
        )

    for file_id in new_test_without_labels["id"]:
        shutil.copyfile(
            src=raw / "images" / f"{file_id}.jpg",
            dst=public / "images" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(new_test_without_labels) == len(
        new_test
    ), "Public and Private tests should have equal length"
    assert len(list(public.glob("images/*.jpg"))) == len(new_train) + len(
        new_test_without_labels
    ), "Public images should have the same number of images as the sum of train and test"

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df[CLASSES] = 1 / len(CLASSES)

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
