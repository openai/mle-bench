import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.competitions.utils import get_logger, rle_encode

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    """
    We make train/test split from old train set, using same train/test proportion as the original
    competition. Concretely, the new split has 18673 train samples and 1856 test samples. We also
    copy over the validation set as-is.

    `sample_submission` is created with random predictions, either "1 3 10 5" or "-" (empty)
    """
    DEV = False

    with open(raw / "train_metadata.json", "r") as f:
        train_metadata = json.load(f)
    train_metadata = train_metadata[:100] if DEV else train_metadata
    with open(raw / "validation_metadata.json", "r") as f:
        validation_metadata = json.load(f)

    if DEV:
        new_train, new_test = train_metadata[:90], train_metadata[90:]
    else:
        new_train, new_test = train_test_split(
            train_metadata, test_size=len(validation_metadata), random_state=0
        )
    logger.info(
        f"Created new split with {len(new_train)} train samples and {len(new_test)} test samples"
    )

    # Create answers csv containing ground truth masks, heights, widths
    test_answers = []
    for sample in tqdm(new_test):
        record_id = sample["record_id"]
        mask = np.load(raw / "train" / record_id / "human_pixel_masks.npy")
        rle = rle_encode(mask)
        rle = " ".join([str(i) for i in rle]) if rle else "-"

        band = np.load(raw / "train" / record_id / "band_08.npy")
        height, width, _ = band.shape
        test_answers.append(
            {
                "record_id": record_id,
                "encoded_pixels": rle,
                "height": height,
                "width": width,
            }
        )
    test_answers = pd.DataFrame(test_answers)

    for sample in tqdm(new_train):
        record_id = sample["record_id"]
        (public / "train" / record_id).mkdir(exist_ok=True, parents=True)
        # Copy bands
        for band_idx in range(8, 17):
            file_name = f"band_{band_idx:02}.npy"
            shutil.copyfile(
                src=raw / "train" / record_id / file_name,
                dst=public / "train" / record_id / file_name,
            )
        # Copy human individual masks
        shutil.copyfile(
            src=raw / "train" / record_id / "human_individual_masks.npy",
            dst=public / "train" / record_id / "human_individual_masks.npy",
        )
        # Copy human pixel masks
        shutil.copyfile(
            src=raw / "train" / record_id / "human_pixel_masks.npy",
            dst=public / "train" / record_id / "human_pixel_masks.npy",
        )

    for sample in tqdm(new_test):
        record_id = sample["record_id"]
        (public / "test" / record_id).mkdir(exist_ok=True, parents=True)
        # Copy bands
        for band_idx in range(8, 17):
            file_name = f"band_{band_idx:02}.npy"
            shutil.copyfile(
                src=raw / "train" / record_id / file_name,
                dst=public / "test" / record_id / file_name,
            )

    # Copy over existing validation data
    (raw / "validation").mkdir(exist_ok=True, parents=True)
    shutil.copytree(raw / "validation", public / "validation", dirs_exist_ok=True)
    shutil.copyfile(raw / "validation_metadata.json", public / "validation_metadata.json")

    # Write other files
    with open(public / "train_metadata.json", "w") as f:
        f.write(json.dumps(new_train))
    test_answers.to_csv(private / "answers.csv", index=False)

    submission_df = test_answers.copy()
    random.seed(0)
    submission_df["encoded_pixels"] = [
        random.choice(["1 3 10 5", "-"]) for _ in range(len(submission_df))
    ]
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # Sanity checks
    assert (public / "train_metadata.json").exists(), "`train_metadata.json` doesn't exist!"
    assert (public / "sample_submission.csv").exists(), "`sample_submission.csv` doesn't exist!"
    assert (
        public / "validation_metadata.json"
    ).exists(), "`validation_metadata.json` doesn't exist!"
    assert (public / "train").exists(), "`train` directory doesn't exist!"
    assert (public / "test").exists(), "`test` directory doesn't exist!"
    assert (public / "validation").exists(), "`public` directory doesn't exist!"
    assert (private / "answers.csv").exists(), "`answers.csv` doesn't exist!"

    new_train_bands = list(img.stem for img in (public / "train").rglob("band*.npy"))
    assert (
        len(new_train_bands) == len(new_train) * 9
    ), f"Expected {len(new_train) * 9} bands in the train set, but got {len(new_train_bands)}!"
    new_test_bands = list(img.stem for img in (public / "test").rglob("band*.npy"))
    assert (
        len(new_test_bands) == len(new_test) * 9
    ), f"Expected {len(new_test) * 9} in the test set, but got {len(new_test_bands)}!"

    new_train_individual_masks = list(
        img.stem for img in (public / "train").rglob("human_individual_masks.npy")
    )
    assert len(new_train_individual_masks) == len(
        new_train
    ), f"Expected 1 human individual mask per sample in the train set, but got {len(new_train_individual_masks)}!"
    new_test_individual_masks = list(
        img.stem for img in (public / "test").rglob("human_individual_masks.npy")
    )
    assert (
        len(new_test_individual_masks) == 0
    ), f"Expected 0 human individual masks per sample in the test set, but got {len(new_test_individual_masks)}!"

    new_train_pixel_masks = list(
        img.stem for img in (public / "train").rglob("human_pixel_masks.npy")
    )
    assert len(new_train_pixel_masks) == len(
        new_train
    ), f"Expected 1 human pixel mask per sample in the train set, but got {len(new_train_pixel_masks)}!"
    new_test_pixel_masks = list(
        img.stem for img in (public / "test").rglob("human_pixel_masks.npy")
    )
    assert (
        len(new_test_pixel_masks) == 0
    ), f"Expected 0 human pixel masks per sample in the test set, but got {len(new_test_pixel_masks)}!"
