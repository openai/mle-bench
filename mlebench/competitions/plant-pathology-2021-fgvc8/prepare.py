import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import extract, read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, answers = train_test_split(old_train, test_size=0.2, random_state=0)

    # Create a sample submission file
    submission_df = answers.copy()
    submission_df["labels"] = "healthy"

    # Checks
    assert len(answers) == len(submission_df), "Answers and submission should have the same length"
    assert not set(new_train["image"]).intersection(
        set(answers["image"])
    ), "new_train and answers should not share any image"
    assert (
        "image" in new_train.columns and "labels" in new_train.columns
    ), "Train DataFrame must have 'image' and 'labels' columns"
    assert (
        "image" in submission_df.columns and "labels" in submission_df.columns
    ), "Sample submission DataFrame must have 'image' and 'labels' columns"
    assert len(new_train) + len(answers) == len(
        old_train
    ), "The combined length of new_train and answers should equal the length of old_train"

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Copy files
    (public / "test_images").mkdir(exist_ok=True)
    (public / "train_images").mkdir(exist_ok=True)

    for file_id in tqdm(new_train["image"], desc="Copying Train Images"):
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}",
            dst=public / "train_images" / f"{file_id}",
        )

    for file_id in tqdm(answers["image"], desc="Copying Test Images"):
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}",
            dst=public / "test_images" / f"{file_id}",
        )

    # Checks
    assert len(list(public.glob("train_images/*.jpg"))) == len(
        new_train
    ), "Public train images should have the same number of images as the train DataFrame"
    assert len(list(public.glob("test_images/*.jpg"))) == len(
        answers
    ), "Public test images should have the same number of images as the answers DataFrame"
