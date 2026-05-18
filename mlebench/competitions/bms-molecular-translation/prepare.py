import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def make_image_subpath(image_id: str) -> Path:
    """
    Creates a triple-nested directory structure from the first 3 characters of the image_id.
    """
    subpath = Path(image_id[0]) / image_id[1] / image_id[2] / f"{image_id}.png"
    return subpath


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Load train data
    old_train = pd.read_csv(raw / "train_labels.csv")

    # Create train, test from train split
    new_train, new_test = train_test_split(old_train, test_size=0.2, random_state=0)
    new_train.to_csv(public / "train_labels.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # Copy train files
    for idx, row in tqdm(new_train.iterrows(), total=len(new_train), desc="Copying train images"):
        image_id = row["image_id"]
        src = raw / "train" / make_image_subpath(image_id)
        dst = public / "train" / make_image_subpath(image_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=src, dst=dst)

    # Copy test files
    for idx, row in tqdm(new_test.iterrows(), total=len(new_test), desc="Copying test images"):
        image_id = row["image_id"]
        src = raw / "train" / make_image_subpath(image_id)
        dst = public / "test" / make_image_subpath(image_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=src, dst=dst)

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["InChI"] = "InChI=1S/H2O/h1H2"
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Copy other files in the dataset (no modification needed)
    shutil.copyfile(src=raw / "extra_approved_InChIs.csv", dst=public / "extra_approved_InChIs.csv")

    # Checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected {len(old_train)} total images in new_train ({len(new_train)}) and new_test ({len(new_test)})"
    assert len(list((public / "train").glob("**/*.png"))) == len(
        new_train
    ), f"Expected {len(new_train)} train images in public/train, but got {len(list((public / 'train').glob('**/*.png')))}"
    assert len(list((public / "test").glob("**/*.png"))) == len(
        new_test
    ), f"Expected {len(new_test)} test images in public/test, but got {len(list((public / 'test').glob('**/*.png')))}"

    assert "image_id" in sample_submission.columns, "Sample submission must have 'image_id' column"
    assert "InChI" in sample_submission.columns, "Sample submission must have 'InChI' column"
    assert len(sample_submission) == len(
        new_test
    ), f"Expected {len(new_test)} images in sample submission, but got {len(sample_submission)}"
