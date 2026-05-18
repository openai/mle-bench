import re
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Data is in train2.zip - we need to unzip it
    shutil.unpack_archive(raw / "train2.zip", raw)

    # Files are named as
    # Train: "YYYYMMDD_HHMMSS_{seconds}_TRAIN{idx}_{label:0,1}.aif"
    # Test: "YYYYMMDD_HHMMSS_{seconds}_Test{idx}.aif"

    # There are 4 days in Train and 3 days in Test
    # In our new dataset, we'll just split Train_old into 2 days for Train and 2 days for Test

    samples_by_date = {}
    n_train_old = 0
    for sample in (raw / "train2").iterdir():
        date = sample.name.split("_")[0]
        if date not in samples_by_date:
            samples_by_date[date] = []
        samples_by_date[date].append(sample)
        n_train_old += 1

    assert len(samples_by_date) == 4, "Expected 4 days in Train_old"
    dates = sorted(list(samples_by_date.keys()))
    new_train = samples_by_date[dates[0]] + samples_by_date[dates[1]]
    new_test = samples_by_date[dates[2]] + samples_by_date[dates[3]]
    # Sort files - filenames have timestamps so we want new idxs to be time-ordered
    new_train = sorted(new_train)
    new_test = sorted(new_test)

    # Copy files to new directories
    (public / "train2").mkdir(exist_ok=True, parents=True)
    for idx, sample in enumerate(tqdm(new_train)):
        # Replace index part of filename with new index
        new_sample_name = re.sub(r"TRAIN\d+", f"TRAIN{idx}", sample.name)
        new_sample = public / "train2" / new_sample_name
        shutil.copy(sample, new_sample)

    answer_rows = []  # While we're at it, collect answers for the new test set
    (public / "test2").mkdir(exist_ok=True, parents=True)
    for idx, sample in enumerate(tqdm(new_test)):
        # Replace everything after the TRAIN{idx} part of the filename
        # (replaces index as well as label part of filename)
        new_sample_name = sample.name.split("TRAIN")[0] + f"Test{idx}.aif"
        new_sample = public / "test2" / new_sample_name
        shutil.copy(sample, new_sample)

        # Add to new test set answers
        answer_rows.append(
            {"clip": new_sample_name, "probability": 1 if sample.stem.endswith("_1") else 0}
        )

    assert len(new_train) == len(
        list((public / "train2").glob("*.aif"))
    ), f"Expected {len(new_train)} samples in new_train ({len(list((public / 'train2').glob('*.aif')))}"
    assert len(new_test) == len(
        list((public / "test2").glob("*.aif"))
    ), f"Expected {len(new_test)} samples in new_test ({len(list((public / 'test2').glob('*.aif')))}"
    assert (
        len(new_train) + len(new_test) == n_train_old
    ), f"Expected {n_train_old} total samples in new_train ({len(new_train)}) and new_test ({len(new_test)})"

    # Make zipped versions
    shutil.make_archive(public / "train2", "zip", public, "train2")
    shutil.make_archive(public / "test2", "zip", public, "test2")
    # Remove unzipped directories (original comp doesn't have these)
    shutil.rmtree(public / "train2")
    shutil.rmtree(public / "test2")
    # we also don't need the raw dirs anymore
    shutil.rmtree(raw / "train2")

    # Create answers
    answers_df = pd.DataFrame(answer_rows)
    answers_df.to_csv(private / "test.csv", index=False)

    # Create sample submission
    sample_submission = answers_df.copy()
    sample_submission["probability"] = 0
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

    assert set(answers_df.columns) == set(
        ["clip", "probability"]
    ), "Answers must have 'clip' and 'probability' columns"
    assert set(sample_submission.columns) == set(
        ["clip", "probability"]
    ), "Sample submission must have 'clip' and 'probability' columns"
