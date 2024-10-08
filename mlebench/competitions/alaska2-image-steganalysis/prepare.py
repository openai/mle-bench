import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # List of all train image IDs
    cover_images_dir = raw / "Cover"
    cover_images = sorted(list(cover_images_dir.glob("*.jpg")))

    # Create train, test from train split
    test_size = 5000
    new_train, new_test = train_test_split(cover_images, test_size=test_size, random_state=42)

    # Prepare dirs
    steganography_algs = ["JMiPOD", "JUNIWARD", "UERD"]
    for subdir in ["Cover", "Test"] + steganography_algs:
        (public / subdir).mkdir(parents=True, exist_ok=True)

    # Copy over new train set, giving them new ids
    for idx, fp in tqdm(enumerate(new_train), total=len(new_train), desc="Copying train images"):
        image_id = idx + 1  # 1-indexed
        shutil.copyfile(src=fp, dst=public / "Cover" / f"{image_id:05d}.jpg")
        for alg in steganography_algs:
            shutil.copyfile(src=raw / alg / fp.name, dst=public / alg / f"{image_id:05d}.jpg")

    # Populate test set
    answers_rows = []
    random.seed(0)
    random.shuffle(new_test)
    for idx, fp in tqdm(enumerate(new_test), total=len(new_test), desc="Copying test images"):
        image_id = idx + 1  # 1-indexed
        test_id = f"{image_id:04d}.jpg"
        dest = public / "Test" / test_id

        # For the test set, we randomly select between the "Cover" (unedited image, negative class)
        # and one of the 3 steganography algorithms (positive class)
        # 1:1 ratio of positive:negative examples, and even distribution of steganography algorithms
        if random.choice([True, False]):
            # Negative class
            shutil.copyfile(
                src=fp,
                dst=dest,
            )
            answers_rows.append({"Id": test_id, "Label": 0})
        else:
            # Positive class
            alg = random.choice(steganography_algs)
            shutil.copyfile(src=raw / alg / fp.name, dst=dest)
            answers_rows.append({"Id": test_id, "Label": 1})

    # Write answers to file
    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(private / "test.csv", index=False)

    # Create sample submission
    sample_submission = answers_df.copy()
    sample_submission["Label"] = 0
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Checks
    assert "Id" in answers_df.columns, "Answers must have 'Id' column"
    assert "Label" in answers_df.columns, "Answers must have 'Label' column"
    assert "Id" in sample_submission.columns, "Sample submission must have 'Id' column"
    assert "Label" in sample_submission.columns, "Sample submission must have 'Label' column"
    assert (
        len(answers_df) == test_size
    ), f"Expected {test_size} test images, but got {len(answers_df)}"
    assert len(sample_submission) == len(
        answers_df
    ), f"Sample submission ({len(sample_submission)}) and answers ({len(answers_df)}) must have the same length"
    assert (
        len(list(public.glob("Test/*.jpg"))) == test_size
    ), f"Expected {test_size} test images in public/Test, but got {len(list(public.glob('Test/*.jpg')))}"
    assert len(list(public.glob("Cover/*.jpg"))) == len(
        new_train
    ), f"Expected {len(new_train)} train images in public/Cover, but got {len(list(public.glob('Cover/*.jpg')))}"
    for alg in steganography_algs:
        assert len(list(public.glob(f"{alg}/*.jpg"))) == len(
            new_train
        ), f"Expected {len(new_train)} train images in public/{alg}, but got {len(list(public.glob(f'{alg}/*.jpg')))}"
