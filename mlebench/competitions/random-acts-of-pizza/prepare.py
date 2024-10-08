import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Load data
    with open(raw / "train.json") as f:
        old_train = json.load(f)
    with open(raw / "test.json") as f:
        old_test = json.load(f)

    test_ratio = len(old_test) / (len(old_train) + len(old_test))

    all_fields = list([key for key in old_train[0].keys()])
    assert all(set(all_fields) == set([key for key in sample.keys()]) for sample in old_train)
    test_fields = list([key for key in old_test[0].keys()])
    assert all(set(test_fields) == set([key for key in sample.keys()]) for sample in old_test)

    # Create train, test from train split
    new_train, new_test = train_test_split(old_train, test_size=test_ratio, random_state=0)

    new_test_without_labels = []
    for sample in new_test:
        # Keep only the fields in the test set
        sample = {key: sample[key] for key in test_fields}
        new_test_without_labels.append(sample)

    # Save the new train and test splits
    with open(public / "train.json", "w") as f:
        json.dump(new_train, f, indent=4)
    with open(public / "test.json", "w") as f:
        json.dump(new_test_without_labels, f, indent=4)
    # Also save zipped versions "train.json.zip" and "test.json.zip"
    shutil.make_archive(public / "train", "zip", public, "train.json")
    shutil.make_archive(public / "test", "zip", public, "test.json")

    # Create answers
    answers_rows = []
    for sample in new_test:
        answers_rows.append(
            {
                "request_id": sample["request_id"],
                "requester_received_pizza": int(sample["requester_received_pizza"]),
            }
        )
    answers = pd.DataFrame(answers_rows)
    answers.to_csv(private / "test.csv", index=False)

    # Create sample submission
    sample_submission = answers.copy()
    sample_submission["requester_received_pizza"] = 0
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

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
        ["request_id", "requester_received_pizza"]
    ), "Answers must have 'request_id' and 'requester_received_pizza' columns"
    assert set(sample_submission.columns) == set(
        ["request_id", "requester_received_pizza"]
    ), "Sample submission must have 'request_id' and 'requester_received_pizza' columns"
