import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def copy(src: Path, dst: Path):
    """A wrapper for `shutil.copy` which creates destination directories when they don't exist."""

    assert src.exists(), f"{src} does not exist"
    assert not dst.exists(), f"{dst} already exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def prepare(raw: Path, public: Path, private: Path):
    old_train = read_csv(raw / "train.csv")
    old_sample_submission = read_csv(raw / "sample_submission.csv")

    # The original dataset has 10,407 train images and 3,469 test images.
    # This implies a 75%/25% train/test split.
    new_train, new_test = train_test_split(old_train, train_size=0.75, random_state=0)

    new_sample_submission = new_test[["image_id", "label"]].copy()
    new_sample_submission["label"] = ""

    new_train.to_csv(public / "train.csv", index=False)
    new_sample_submission.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    for row in new_train.itertuples():
        copy(
            raw / "train_images" / row.label / row.image_id,
            public / "train_images" / row.label / row.image_id,
        )

    for row in new_test.itertuples():
        copy(
            raw / "train_images" / row.label / row.image_id,
            public / "test_images" / row.image_id,
        )

    # Sanity checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected the combined size of the new train and test sets to be the same size as the original train set, but it wasn't! Got {len(new_train) + len(new_test)} != {len(old_train)}."

    new_train_image_ids = set(new_train.image_id)
    new_test_image_ids = set(new_test.image_id)

    assert new_train_image_ids.isdisjoint(new_test_image_ids), "Train and test sets overlap!"

    assert set(new_train.columns) == set(
        old_train.columns
    ), f"Expected the new train set to have the same columns as the original train set, but it didn't! Got {set(new_train.columns)} != {set(old_train.columns)}."

    assert set(new_sample_submission.columns) == set(
        old_sample_submission.columns
    ), f"Expected the new sample submission to have the same columns as the original sample submission, but it didn't! Got {set(new_sample_submission.columns)} != {set(old_sample_submission.columns)}."

    assert len(list(public.glob("train_images/*/*.jpg"))) == len(
        new_train
    ), f"Expected the number of images in the `public / train_images` directory to match the number of rows in the `public / train.csv` file, but it didn't! Got {len(list(public.glob('train_images/*/*.jpg')))} != {len(new_train)}."

    assert len(list(public.glob("test_images/*.jpg"))) == len(
        new_test
    ), f"Expected the number of images in the `public / test_images` directory to match the number of rows in the `private / test.csv` file, but it didn't! Got {len(list(public.glob('test_images/*.jpg')))} != {len(new_test)}."
