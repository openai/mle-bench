import random
import shutil
from pathlib import Path

import pandas as pd


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.

    Raw dataset has:
    - Train: training1, training2, training3, training4
    - Val: validation1, validation2, validation3 (no labels)
    - Test: (not available)

    New prepared dataset has:
    - Train: training1, training2, training3
    - Val: validation1, validation2, validation3 (no labels)
    - Test: training4 (renamed to `test.tar.gz`)

    Furthermore,
    - We modify the `training.csv` to remove training4 samples
    - We modify the `test.csv` and `randomPredictions.csv` to include only training4 IDs

    No other changes. We copy over the remaining files (devel01-40.7z, valid_all_files_combined.7z, sample_code_mmrgc.zip) as-is.
    """

    # Unzip the training4 file to get new test IDs
    shutil.unpack_archive(raw / "training4.tar.gz", raw / "training4")
    # training4 contains samples like "Sample00300.zip", the ID is the last 4 digits ("0300")
    test_ids = sorted([fp.stem[-4:] for fp in (raw / "training4").glob("*.zip")])

    # Update training.csv to remove training4 samples
    training_df = pd.read_csv(raw / "training.csv", dtype={"Id": str, "Sequence": str})
    new_training_df = training_df[~training_df["Id"].isin(test_ids)]
    new_training_df.to_csv(public / "training.csv", index=False)
    assert len(new_training_df) == len(training_df) - len(
        test_ids
    ), f"Expected {len(training_df) - len(test_ids)} samples in training.csv, but got {len(new_training_df)}"

    # Make private answers
    answers_df = training_df[training_df["Id"].isin(test_ids)]
    answers_df.to_csv(private / "test.csv", index=False)
    assert len(answers_df) == len(
        test_ids
    ), f"Expected {len(test_ids)} samples in private/test.csv, but got {len(answers_df)}"

    # Make new public test.csv
    test_df = pd.DataFrame({"Id": test_ids})
    test_df.to_csv(public / "test.csv", index=False)
    assert len(test_df) == len(
        test_ids
    ), f"Expected {len(test_ids)} samples in public/test.csv, but got {len(test_df)}"

    # Make new public randomPredictions.csv
    # predictions are random shufflings of numbers 1-20 (no repeats)
    random.seed(0)
    preds = []
    for _ in range(len(test_ids)):
        pred = " ".join(str(x) for x in random.sample(range(1, 21), 20))
        preds.append(pred)
    random_predictions_df = pd.DataFrame({"Id": test_ids, "Sequence": preds})
    random_predictions_df.to_csv(public / "randomPredictions.csv", index=False)
    assert len(random_predictions_df) == len(
        test_ids
    ), f"Expected {len(test_ids)} samples in public/randomPredictions.csv, but got {len(random_predictions_df)}"

    # Copy over training4 as new test set
    shutil.copyfile(src=raw / "training4.tar.gz", dst=public / "test.tar.gz")

    # Copy over train and validation tars
    for file in [
        "training1.tar.gz",
        "training2.tar.gz",
        "training3.tar.gz",
        "validation1.tar.gz",
        "validation2.tar.gz",
        "validation3.tar.gz",
    ]:
        shutil.copyfile(src=raw / file, dst=public / file)

    # Copy over the rest of the files
    for file in ["devel01-40.7z", "valid_all_files_combined.7z", "sample_code_mmrgc.zip"]:
        shutil.copyfile(src=raw / file, dst=public / file)
