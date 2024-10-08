import shutil
from pathlib import Path

import py7zr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import extract, read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # extract only what we need
    extract(raw / "train.7z", raw)
    extract(raw / "train_labels.csv.zip", raw)

    # Create train, test from train split
    # Original ratio is 1531/(1531+2295) = 0.4
    test_ratio = 0.2
    old_train = read_csv(raw / "train_labels.csv")
    new_train, answers = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Sample submission
    sample_submission = answers.copy()
    sample_submission["invasive"] = 0.5

    # Checks
    assert new_train["name"].is_unique, "new_train should have unique names"
    assert answers["name"].is_unique, "answers should have unique names"
    assert set(new_train["name"]).isdisjoint(
        set(answers["name"])
    ), "new_train and answers should be disjoint"
    assert len(new_train) + len(answers) == len(
        old_train
    ), "new_train and answers together should have the same number of rows as old_train"
    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), "new_train should have the same columns as old_train"
    assert (
        answers.columns.tolist() == old_train.columns.tolist()
    ), "answers should have the same columns as old_train"
    assert (
        sample_submission.columns.tolist() == old_train.columns.tolist()
    ), "sample_submission should have the same columns as old_train"

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train_labels.csv", index=False)
    sample_submission.to_csv(private / "sample_submission.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Copy files
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in tqdm(new_train["name"], desc="Copying Train Images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    for file_id in tqdm(answers["name"], desc="Copying Test Images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    # Checks
    assert len(list((public / "train").glob("*.jpg"))) == len(
        new_train
    ), "public/train should have the same number of files as new_train"
    assert len(list((public / "test").glob("*.jpg"))) == len(
        answers
    ), "public/test should have the same number of files as answers"

    # Zip
    shutil.make_archive(
        str(public / "sample_submission.csv"),
        "zip",
        root_dir=public,
        base_dir="sample_submission.csv",
    )
    shutil.make_archive(
        str(public / "train_labels.csv"), "zip", root_dir=public, base_dir="train_labels.csv"
    )
    with py7zr.SevenZipFile(public / "train.7z", "w") as z:
        z.write(public / "train")
    with py7zr.SevenZipFile(public / "test.7z", "w") as z:
        z.write(public / "test")

    # Delete
    shutil.rmtree(public / "train")
    shutil.rmtree(public / "test")
    (public / "sample_submission.csv").unlink()
    (public / "train_labels.csv").unlink()
