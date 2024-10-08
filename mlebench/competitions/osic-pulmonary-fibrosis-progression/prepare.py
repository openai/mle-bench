import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def copy_dir(args):
    src_dir, dst_dir = args
    shutil.copytree(src=src_dir, dst=dst_dir, dirs_exist_ok=True)


def prepare(raw: Path, public: Path, private: Path):
    # Splitting the train set into train and test with unique Patients
    old_train = read_csv(raw / "train.csv")
    grouped_by_patient = list(old_train.groupby("Patient"))
    train_groups, test_groups = train_test_split(grouped_by_patient, test_size=0.1, random_state=0)
    new_train = pd.concat([group for _, group in train_groups])
    new_test = pd.concat([group for _, group in test_groups])
    assert set(new_train["Patient"]).isdisjoint(
        set(new_test["Patient"])
    ), "There are Patients who are in both train and test sets."

    # For the public new_test set we will only keep each patients first FVS measurement. The task is to predict FVS measurements for all possible weeks
    new_test_public = new_test.sort_values(by="Weeks").groupby("Patient").first().reset_index()

    # Creating the private answers CSV. We need to fill out dummy FVS measurements for all weeks that don't have data so as to match sample_submission.csv
    # Create a DataFrame with all possible Patient-Week combinations
    all_weeks = pd.DataFrame(
        [
            (patient, week)
            for patient in new_test["Patient"].unique()
            for week in range(new_test["Weeks"].min(), new_test["Weeks"].max() + 1)
        ],
        columns=["Patient", "Weeks"],
    )
    # Merge with the new_test DataFrame to fill in missing weeks with NaN values
    new_test = all_weeks.merge(new_test, on=["Patient", "Weeks"], how="left")
    new_test["Patient_Week"] = new_test["Patient"] + "_" + new_test["Weeks"].astype(str)
    new_test["Confidence"] = 100
    assert (
        new_test.groupby("Patient").size().nunique() == 1
    ), "Not all patients have the same number of rows."

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df = submission_df[["Patient_Week"]]
    submission_df["FVC"] = 2000  # Dummy predictions
    submission_df["Confidence"] = 100  # Dummy confidence

    # Write CSVs
    new_train.to_csv(public / "train.csv", index=False)
    new_test_public.to_csv(public / "test.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Copy over data files
    (public / "train").mkdir(exist_ok=True)
    train_args = [
        (raw / "train" / patient, public / "train" / patient)
        for patient in new_train["Patient"].unique()
    ]
    with Pool() as pool:
        list(
            tqdm(pool.imap(copy_dir, train_args), total=len(train_args), desc="Copying train data")
        )

    (public / "test").mkdir(exist_ok=True)
    test_args = [
        (raw / "train" / patient, public / "test" / patient)
        for patient in new_test["Patient"].unique()
    ]
    with Pool() as pool:
        list(tqdm(pool.imap(copy_dir, test_args), total=len(test_args), desc="Copying test data"))

    # Final checks
    assert new_train.shape[1] == 7, f"Expected 7 columns in new_train, but got {new_train.shape[1]}"
    assert new_test.shape[1] == 9, f"Expected 9 columns in new_test, but got {new_test.shape[1]}"
    assert (
        new_test_public.shape[1] == 7
    ), f"Expected 7 columns in new_test_public, but got {new_test_public.shape[1]}"
    assert (
        submission_df.shape[1] == 3
    ), f"Expected 3 columns in submission_df, but got {submission_df.shape[1]}"

    public_train_dirs = set((public / "train").iterdir())
    public_test_dirs = set((public / "test").iterdir())
    common_dirs = public_train_dirs.intersection(public_test_dirs)
    assert (
        not common_dirs
    ), f"There are directories with the same name in public train and test: {common_dirs}"
