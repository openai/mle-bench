import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    submission_df = answers.copy()
    submission_df["time_to_eruption"] = 0

    # Checks
    assert len(answers) == len(submission_df), "Answers and submission should have the same length"
    assert not any(
        new_train["segment_id"].isin(answers["segment_id"])
    ), "No segment_id should be in both train and answers"
    assert list(new_train.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "new_train should have columns 'segment_id' and 'time_to_eruption'"
    assert list(submission_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "submission_df should have columns 'segment_id' and 'time_to_eruption'"
    assert list(answers.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "answers should have columns 'segment_id' and 'time_to_eruption'"
    assert len(new_train) + len(answers) == len(
        old_train
    ), "The sum of the length of new_train and answers should be equal to the length of old_train"

    # Write CSVs
    answers.to_csv(private / "test.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Copy over files
    (public / "test").mkdir(exist_ok=True)
    (public / "train").mkdir(exist_ok=True)

    for file_id in tqdm(new_train["segment_id"], desc="Copying train files"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.csv",
            dst=public / "train" / f"{file_id}.csv",
        )

    for file_id in tqdm(answers["segment_id"], desc="Copying test files"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.csv",
            dst=public / "test" / f"{file_id}.csv",
        )

    # Checks on files
    assert len(list(public.glob("train/*.csv"))) == len(
        new_train
    ), "Public train should have the same number of files as new_train"
    assert len(list(public.glob("test/*.csv"))) == len(
        answers
    ), "Public test should have the same number of files as answers"
