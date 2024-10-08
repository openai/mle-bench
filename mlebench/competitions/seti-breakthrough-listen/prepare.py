import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "train_labels.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Copy over files
    new_train.to_csv(public / "train_labels.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    shutil.copytree(raw / "old_leaky_data", public / "old_leaky_data", dirs_exist_ok=True)

    for file_id in tqdm(new_train["id"], desc="Copying train files"):
        subdir = file_id[0]
        src = raw / "train" / subdir / f"{file_id}.npy"
        dst = public / "train" / subdir / f"{file_id}.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    for file_id in tqdm(new_test["id"], desc="Copying test files"):
        subdir = file_id[0]
        src = raw / "train" / subdir / f"{file_id}.npy"
        dst = public / "test" / subdir / f"{file_id}.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["target"] = 0.5  # Overwrite with dummy values
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Checks
    assert len(sample_submission) == len(
        new_test
    ), "Sample submission length does not match test length."
    assert not set(new_train["id"]).intersection(
        set(new_test["id"])
    ), "There are overlapping IDs in train and test sets."

    train_files = {
        file_path.name: file_path
        for file_path in (public / "train").rglob("*")
        if file_path.is_file()
    }
    test_files = {
        file_path.name: file_path
        for file_path in (public / "test").rglob("*")
        if file_path.is_file()
    }

    assert len(train_files) == len(
        new_train
    ), "Number of train files does not match the number of train records."
    assert len(test_files) == len(
        new_test
    ), "Number of test files does not match the number of test records."
    assert train_files.keys().isdisjoint(
        test_files.keys()
    ), "There are overlapping files in train and test directories."
    assert len(sample_submission) == len(
        new_test
    ), "Sample submission length does not match new test length."
