import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import compress, extract, read_csv


def prepare(raw: Path, public: Path, private: Path):
    extract(raw / "train.zip", raw)
    extract(raw / "test.zip", raw)

    all_train_images = sorted(list((raw / "train").glob("*.jpg")))
    # Original test ratio has Train set - 25,000 samples; Test set - 12,500 samples (33% ratio)
    # We use 0.1 ratio to avoid removing too many samples from train
    train_images, test_images = train_test_split(all_train_images, test_size=0.1, random_state=0)

    # Copy over train images. Rename cat files to cat.0.jpg, cat.1.jpg, etc.
    # Rename dog files to dog.0.jpg, dog.1.jpg, etc.
    cat_ctr = 0
    dog_ctr = 0
    (public / "train").mkdir(exist_ok=True)
    for img in tqdm(train_images):
        if "cat" in img.name:
            shutil.copy(img, public / "train" / f"cat.{cat_ctr}.jpg")
            cat_ctr += 1
        else:
            shutil.copy(img, public / "train" / f"dog.{dog_ctr}.jpg")
            dog_ctr += 1
    assert cat_ctr + dog_ctr == len(
        train_images
    ), f"Expected {len(train_images)} train images but got {cat_ctr + dog_ctr} images."

    # Copy over test images. Rename files to 1.jpg, 2.jpg, etc.
    (public / "test").mkdir(exist_ok=True)
    for i, img in enumerate(tqdm(test_images), start=1):
        shutil.copy(img, public / "test" / f"{i}.jpg")
    assert i == len(test_images), f"Expected {len(test_images)} test images but got {i} images."

    # Compress train and test images, then remove the raw images
    compress(public / "train", public / "train.zip", exist_ok=True)
    compress(public / "test", public / "test.zip", exist_ok=True)
    shutil.rmtree(raw / "train")
    shutil.rmtree(raw / "test")

    # Make answers
    answers = pd.DataFrame(
        {
            "id": [i for i in range(1, len(test_images) + 1)],
            "label": [int("dog" in img.name) for img in test_images],
        }
    )
    answers.to_csv(private / "answers.csv", index=False)
    assert len(answers) == len(
        test_images
    ), f"Expected {len(test_images)} answers but got {len(answers)} answers."

    # Make sample submission
    sample_submission = pd.DataFrame(
        {
            "id": [i for i in range(1, len(test_images) + 1)],
            "label": [0.5 for _ in range(1, len(test_images) + 1)],
        }
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    assert len(sample_submission) == len(
        test_images
    ), f"Expected {len(test_images)} sample submission rows but got {len(sample_submission)} rows."
