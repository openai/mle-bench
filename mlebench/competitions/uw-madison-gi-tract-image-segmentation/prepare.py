import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv


def verify_directory_sync(df: pd.DataFrame, expected_dir: Path, unexpected_dir: Path):
    """
    Checks that the dataframe contents match the directory structure.
    """
    for _, row in tqdm(
        df.iterrows(), desc=f"Verifying directory sync for {expected_dir.name}", total=len(df)
    ):
        case_day_path = expected_dir / row["case"] / f"{row['case']}_{row['day']}"
        assert (
            case_day_path.exists()
        ), f"Directory {case_day_path} does not exist but is listed in the dataframe."
        non_existent_path = unexpected_dir / row["case"] / f"{row['case']}_{row['day']}"
        assert (
            not non_existent_path.exists()
        ), f"Directory {non_existent_path} exists but is not listed in the dataframe."


def prepare(raw: Path, public: Path, private: Path):
    old_train = read_csv(raw / "train.csv")

    # ----------------------- Splitting
    # Extract case and day from 'id'
    old_train["case"] = old_train["id"].apply(lambda x: x.split("_")[0])
    old_train["day"] = old_train["id"].apply(lambda x: x.split("_")[1])
    old_train["slice"] = old_train["id"].apply(lambda x: x.split("_")[-1])

    # Split cases into train and test
    unique_cases = old_train["case"].unique()
    train_cases, test_cases = train_test_split(unique_cases, test_size=0.1, random_state=42)

    # Initially assign entire cases to train or test set
    old_train["set"] = old_train["case"].apply(lambda x: "test" if x in test_cases else "train")

    # Then mark some days from train to be test, to match competition test description
    days_df = old_train[old_train["set"] == "train"].groupby("case")["day"].apply(set).reset_index()
    for _, row in days_df.iterrows():
        # if theres more than 4 days, we will move any days past the 4th to the test set
        days = row["day"]
        if len(days) > 4:
            days = sorted(days, key=lambda x: int(x[len("day") :]))
            days_to_move = days[4:]
            # change their set to "test"
            old_train.loc[
                old_train["case"].eq(row["case"]) & old_train["day"].isin(days_to_move), "set"
            ] = "test"

    # ----------------------- Move the files to the correct new locations
    old_train_dir = raw / "train"
    new_train_dir = public / "train"
    new_test_dir = public / "test"

    # Create new directories if they don't exist
    new_train_dir.mkdir(parents=True, exist_ok=True)
    new_test_dir.mkdir(parents=True, exist_ok=True)

    # Move directories based on the set assignment
    for case in tqdm(unique_cases, desc="Splitting by case"):
        original_path = old_train_dir / case
        if case in train_cases:
            new_path = new_train_dir / case
        else:
            new_path = new_test_dir / case
        # new_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(original_path, new_path, dirs_exist_ok=True)

    # Move specific days from public/train/ to public/test/ for marked case-days
    for _, row in tqdm(
        old_train.iterrows(), desc="Handling additional day-based splits", total=len(old_train)
    ):
        if row["set"] == "test":
            source_day_path = new_train_dir / row["case"] / f"{row['case']}_{row['day']}"
            target_day_path = new_test_dir / row["case"] / f"{row['case']}_{row['day']}"
            if source_day_path.exists():
                target_day_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source_day_path.as_posix(), target_day_path.as_posix())

    # ------------------------ Saving splits
    new_train = old_train[old_train["set"] == "train"].copy()
    new_test = old_train[old_train["set"] == "test"].copy()
    # some asserts before we drop columns
    verify_directory_sync(new_train, expected_dir=new_train_dir, unexpected_dir=new_test_dir)
    verify_directory_sync(new_test, expected_dir=new_test_dir, unexpected_dir=new_train_dir)

    # get image height and image width for the test set, since this is needed for the metric
    for _, row in tqdm(
        new_test.iterrows(), desc="Getting image dimensions for test set", total=len(new_test)
    ):
        case, day, day_slice = row["case"], row["day"], row["slice"]
        image_paths = list(
            (old_train_dir / case / f"{case}_{day}" / "scans").glob(f"slice_{day_slice}_*.png")
        )
        assert len(image_paths) == 1, f"Expected 1 image, found {len(image_paths)}"
        image_path = image_paths[0]
        width, height = (int(length) for length in image_path.stem.split("_")[2:4])
        new_test.loc[row.name, "image_width"] = width
        new_test.loc[row.name, "image_height"] = height

    # dont need these anymore, and werent part of the original data
    new_train.drop(columns=["set", "case", "day", "slice"], inplace=True)
    new_test.drop(columns=["set", "case", "day", "slice"], inplace=True)

    # create sample submission
    sample_submission = new_test.copy()
    sample_submission["segmentation"] = "1 1 5 2"
    # these are just metadata for the private test set necessary for the metric
    sample_submission.drop(columns=["image_height", "image_width"], inplace=True)
    # rename 'segmentation' to 'predicted' to match kaggle.com
    sample_submission.rename(columns={"segmentation": "predicted"}, inplace=True)
    sample_submission.to_csv(public / "sample_submission.csv", index=False, na_rep="")

    # create private files
    # rename 'segmentation' to 'predicted' to match sample_submission format
    new_test.rename(columns={"segmentation": "predicted"}, inplace=True)
    new_test.to_csv(private / "test.csv", index=False, na_rep="")

    # create public files
    new_train.to_csv(public / "train.csv", index=False, na_rep="")
    # including this because we are converting this from code to csv competition
    # and we need to point the model to the ids it needs to produce labels for
    new_test_without_labels = new_test.drop(columns=["predicted", "image_width", "image_height"])
    new_test_without_labels.to_csv(public / "test.csv", index=False, na_rep="")

    # ------------------------ checks

    assert new_test_without_labels.shape[1] == 2, "Public test should have 2 columns."
    assert new_train.shape[1] == 3, "Public train should have 3 columns."
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Train and test should sum up to the original data."
