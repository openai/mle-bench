import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    dev = False

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    # 25958/(142119+ 25958) = 0.15 original test split
    new_train, answers = train_test_split(old_train, test_size=0.15, random_state=0)

    # Sample submission
    sample_submission_df = answers.copy()
    sample_submission_df["attribute_ids"] = "0 1 2"

    # Checks
    assert len(answers) + len(new_train) == len(
        old_train
    ), "New train and test should have the same length as the old train"
    assert len(answers) == len(
        sample_submission_df
    ), "Answers and sample submission should have the same length"
    assert (
        answers.columns.tolist() == old_train.columns.tolist()
    ), "Answers and old_train should have the same columns"
    assert (
        sample_submission_df.columns.tolist() == old_train.columns.tolist()
    ), "Sample submission and old train should have the same columns"
    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), "New train and old train should have the same columns"
    assert set(new_train["id"]).isdisjoint(
        set(answers["id"])
    ), "New train and answers should have no overlap of rows"

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Copy over files
    shutil.copyfile(src=raw / "labels.csv", dst=public / "labels.csv")

    (public / "test").mkdir(exist_ok=True)
    (public / "train").mkdir(exist_ok=True)

    if dev:
        # Reduce the number of images to be copied to 10% of the original, for testing
        new_train = new_train.head(int(len(new_train) * 0.1))
        answers = answers.head(int(len(answers) * 0.1))

    for file_id in tqdm(new_train["id"], desc="Copying train images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.png",
            dst=public / "train" / f"{file_id}.png",
        )

    for file_id in tqdm(answers["id"], desc="Copying test images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.png",
            dst=public / "test" / f"{file_id}.png",
        )

    # File copying checks
    assert len(list(public.glob("train/*.png"))) == len(
        new_train
    ), "Train images should have the same number of images as the train"
    assert len(list(public.glob("test/*.png"))) == len(
        answers
    ), "Test images should have the same number of images as the test"
