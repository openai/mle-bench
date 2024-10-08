import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import extract, read_csv


def create_dataframe_from_directory(directory: str) -> pd.DataFrame:
    """
    Creates a DataFrame from a directory of images.

    Args:
        directory (str): The path to the directory containing subdirectories of images.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'image' and 'label'. The 'image' column contains the file paths to the images, and the 'label' column contains the corresponding labels (subdirectory names).
    """
    data = []
    for label in sorted(os.listdir(directory)):  # Sort labels for determinism
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for file_name in sorted(os.listdir(label_path)):  # Sort files for determinism
                if file_name.endswith(".png"):
                    file_path = os.path.join(label_path, file_name)
                    data.append({"file": os.path.basename(file_path), "species": label})
    return pd.DataFrame(data)


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Directory containing the training images
    train_dir = raw / "train"
    old_train = create_dataframe_from_directory(train_dir)
    test_ratio = 0.14  # 794/(4750+794) = 0.14
    train_df, test_df = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Create a sample submission file
    submission_df = test_df.copy()
    submission_df["species"] = "Sugar beet"

    # Checks
    assert len(test_df) == len(submission_df), "Answers and submission should have the same length"
    assert not set(train_df["file"]).intersection(
        set(test_df["file"])
    ), "new_train and answers should not share any image"
    assert (
        "file" in train_df.columns and "species" in train_df.columns
    ), "Train DataFrame must have 'file' and 'species' columns"
    assert (
        "file" in submission_df.columns and "species" in submission_df.columns
    ), "Sample submission DataFrame must have 'file' and 'species' columns"
    assert len(train_df) + len(test_df) == len(
        old_train
    ), "The combined length of new_train and answers should equal the length of old_train"

    # Write CSVs
    test_df.to_csv(private / "answers.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Copy files
    (public / "test").mkdir(exist_ok=True)
    (public / "train").mkdir(exist_ok=True)

    # Create nested folder structure for train
    for species in train_df["species"].unique():
        (public / "train" / species).mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(train_df.iterrows(), desc="Copying Train Images", total=len(train_df)):
        src_path = train_dir / row["species"] / row["file"]
        dst_path = public / "train" / row["species"] / row["file"]
        shutil.copyfile(src=src_path, dst=dst_path)

    for _, row in tqdm(test_df.iterrows(), desc="Copying Test Images", total=len(test_df)):
        src_path = train_dir / row["species"] / row["file"]
        dst_path = public / "test" / row["file"]
        shutil.copyfile(src=src_path, dst=dst_path)

    # Checks
    assert len(list(public.glob("train/**/*.png"))) == len(
        train_df
    ), f"Public train images should have the same number of images as the train DataFrame: number of files {len(list(public.glob('train/**/*.png')))} != len(train_df)={len(train_df)}"
    assert len(list(public.glob("test/*.png"))) == len(
        test_df
    ), f"Public test images should have the same number of images as the answers DataFrame: number of files {len(list(public.glob('test/*.png')))} != len(test_df)={len(test_df)}"
