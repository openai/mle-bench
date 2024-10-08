import json
import shutil
from pathlib import Path

import pandas as pd
import py7zr
from sklearn.model_selection import train_test_split

from mlebench.utils import extract


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    extract(raw / "train.json.7z", raw)
    extract(raw / "test.json.7z", raw)
    old_train = json.load((raw / "data/processed/train.json").open())
    old_test = json.load((raw / "data/processed/test.json").open())

    all_fields = list([key for key in old_train[0].keys()])
    assert all(
        set(all_fields) == set([key for key in sample.keys()]) for sample in old_train
    ), "Inconsistent fields in train set"
    test_fields = list([key for key in old_test[0].keys()])
    assert all(
        set(test_fields) == set([key for key in sample.keys()]) for sample in old_test
    ), "Inconsistent fields in test set"

    # Old ratio is Train set - 1,604 samples; Test set - 8,424 samples (~84% ratio)
    # We do a 20% ratio to avoid removing too many samples from train
    new_train, new_test = train_test_split(old_train, test_size=0.2, random_state=0)
    new_test_without_labels = []
    for sample in new_test:
        # Keep only the fields in the test set
        sample = {key: sample[key] for key in test_fields}
        new_test_without_labels.append(sample)

    # Write new train and test splits, compress, then remove the uncompressed files
    (private / "tmp_data").mkdir(exist_ok=True)
    with open(private / "tmp_data" / "train.json", "w") as f:
        json.dump(new_train, f)
    with open(private / "tmp_data" / "test.json", "w") as f:
        json.dump(new_test_without_labels, f)

    with py7zr.SevenZipFile(public / "train.json.7z", "w") as archive:
        archive.write(
            private / "tmp_data" / "train.json",
            arcname=(private / "tmp_data" / "train.json").relative_to(private / "tmp_data"),
        )

    with py7zr.SevenZipFile(public / "test.json.7z", "w") as archive:
        archive.write(
            private / "tmp_data" / "test.json",
            arcname=(private / "tmp_data" / "test.json").relative_to(private / "tmp_data"),
        )

    # Make answers as csv from json
    answer_rows = []
    for sample in new_test:
        answer_rows.append(
            {
                "id": sample["id"],
                "is_iceberg": int(sample["is_iceberg"]),
            }
        )
    answers = pd.DataFrame(answer_rows)
    answers.to_csv(private / "test.csv", index=False)

    # Make sample submission
    sample_submission = answers.copy()
    sample_submission["is_iceberg"] = 0.5
    sample_submission.to_csv(private / "sample_submission.csv", index=False)
    with py7zr.SevenZipFile(public / "sample_submission.csv.7z", "w") as archive:
        archive.write(
            private / "sample_submission.csv",
            arcname=(private / "sample_submission.csv").relative_to(private),
        )

    # Remove uncompressed files
    shutil.rmtree(private / "tmp_data")

    # Checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected {len(old_train)} total samples in new_train ({len(new_train)}) and new_test ({len(new_test)})"
    assert len(new_test) == len(
        new_test_without_labels
    ), f"Expected new_test ({len(new_test)}) to have the same length as new_test_without_labels ({len(new_test_without_labels)})"
    assert len(answers) == len(
        new_test
    ), f"Expected answers ({len(answers)}) to have the same length as new_test ({len(new_test)})"
    assert len(sample_submission) == len(
        new_test
    ), f"Expected sample_submission ({len(sample_submission)}) to have the same length as new_test ({len(new_test)})"
    assert set(answers.columns) == set(
        ["id", "is_iceberg"]
    ), "Answers must have 'id' and 'is_iceberg' columns"
    assert set(sample_submission.columns) == set(
        ["id", "is_iceberg"]
    ), "Sample submission must have 'id' and 'is_iceberg' columns"

    new_train_ids = set([sample["id"] for sample in new_train])
    new_test_ids = set([sample["id"] for sample in new_test])
    assert new_train_ids.isdisjoint(new_test_ids), "Train and test ids should not overlap"
