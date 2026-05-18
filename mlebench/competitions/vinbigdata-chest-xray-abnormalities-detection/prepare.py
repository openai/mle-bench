import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    dev = False

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    unique_image_ids = old_train["image_id"].unique()
    # Original train has 15k images, original test has 3k images
    # Our new train will have 13.5k images, our new test will have 1.5k images
    expected_train_size = 13500
    expected_test_size = 1500
    train_image_ids, test_image_ids = train_test_split(
        unique_image_ids, test_size=0.1, random_state=0
    )

    new_train = old_train[old_train["image_id"].isin(train_image_ids)]
    answers = old_train[old_train["image_id"].isin(test_image_ids)]

    # Create sample submission
    sample_submission = pd.DataFrame(
        {
            "image_id": test_image_ids,
            "PredictionString": "14 1 0 0 1 1",
        }  # As per the original sample submission
    )

    # Checks
    assert (
        len(set(new_train["image_id"])) == expected_train_size
    ), f"Expected {expected_train_size} train image_ids, got {len(set(new_train['image_id']))}"
    assert (
        len(set(answers["image_id"])) == expected_test_size
    ), f"Expected {expected_test_size} test image_ids, got {len(set(answers['image_id']))}"
    assert set(new_train["image_id"]).isdisjoint(
        set(answers["image_id"])
    ), f"image_id is not disjoint between train and test sets"
    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), f"Columns of new train and old train are not the same: {new_train.columns.tolist()} vs {old_train.columns.tolist()}"
    assert len(new_train) + len(answers) == len(
        old_train
    ), f"Length of new train and answers should add up to the length of old train, got {len(new_train) + len(answers)} vs {len(old_train)}"
    assert len(sample_submission) == len(
        set(answers["image_id"])
    ), f"Length of sample submission should be equal to the number of unique image_ids in answers, got {len(sample_submission)} vs {len(set(answers['image_id']))}"

    # Reformat answers
    def _get_consensus_annotation(answers, inspect_duplicates=False):
        """
        In the original train, there can be multiple annotations of the same image_id, class_id pair. (Different radiologists draw the bounding boxes differently for the same finding)

        In the original test, there is only one annotation per image_id, class_id pair. The original test set is labeled by consensus of 5 radiologists. (Source: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/207969#1134645)

        We simulate consensus by taking the first annotation for each image_id, class_id pair.
        """

        if inspect_duplicates:
            duplicates = answers[answers.duplicated(subset=["image_id", "class_id"], keep=False)]
            duplicates = duplicates.sort_values(by=["image_id", "class_id"])
            duplicates.to_csv("duplicates.csv", index=False)

        answers = answers.groupby(by=["image_id", "class_id"]).first().reset_index()
        return answers

    answers = _get_consensus_annotation(answers)
    # Filling in missing values for when there is no finding (class_id = 14)
    answers = answers.fillna(0)
    answers.loc[answers["class_id"] == 14, ["x_max", "y_max"]] = 1.0

    # Create gold submission
    gold = answers[["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]].copy()
    # Create individual prediction strings
    gold.loc[:, "PredictionString"] = gold.apply(
        lambda row: f"{row['class_id']} 1.0 {row['x_min']} {row['y_min']} {row['x_max']} {row['y_max']}",
        axis=1,  # 1.0 is the confidence score
    )
    # Group by image_id and concatenate prediction strings
    gold = gold.groupby("image_id")["PredictionString"].agg(" ".join).reset_index()
    gold = gold.reset_index(drop=True)
    assert len(gold) == len(
        set(answers["image_id"])
    ), f"Length of gold should be equal to the number of unique image_ids in answers, got {len(gold)} vs {len(set(answers['image_id']))}"

    # Write CSVs
    new_train.to_csv(public / "train.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)
    gold.to_csv(private / "gold_submission.csv", index=False)

    # Copy over files
    (public / "test").mkdir(exist_ok=True)
    (public / "train").mkdir(exist_ok=True)

    if dev == True:
        train_image_ids = train_image_ids[:10]
        test_image_ids = test_image_ids[:10]

    for file_id in tqdm(train_image_ids, desc="Copying train files"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.dicom",
            dst=public / "train" / f"{file_id}.dicom",
        )

    for file_id in tqdm(test_image_ids, desc="Copying test files"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.dicom",
            dst=public / "test" / f"{file_id}.dicom",
        )

    # Check files
    assert len(list(public.glob("train/*.dicom"))) == len(
        train_image_ids
    ), f"Expected {len(train_image_ids)} train files, got {len(list(public.glob('train/*.dicom')))}"
    assert len(list(public.glob("test/*.dicom"))) == len(
        test_image_ids
    ), f"Expected {len(test_image_ids)} test files, got {len(list(public.glob('test/*.dicom')))}"
