from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Create train/test split from old train set, using same train/test proportion
    """
    old_train = np.load(raw / "timit_11" / "timit_11" / "train_11.npy")
    old_train_label = np.load(raw / "timit_11" / "timit_11" / "train_label_11.npy")
    old_train_idxs = range(len(old_train))

    # Create new splits
    old_test = np.load(raw / "timit_11" / "timit_11" / "test_11.npy")
    old_test_prop = len(old_test) / (len(old_train) + len(old_test))  # ~= 0.268
    new_train_idxs, new_test_idxs = train_test_split(
        old_train_idxs, test_size=old_test_prop, random_state=0
    )

    new_train = old_train[new_train_idxs]
    new_train_label = old_train_label[new_train_idxs]
    new_test = old_train[new_test_idxs]
    new_test_label = old_train_label[new_test_idxs]

    answers_df = pd.DataFrame({"Id": range(len(new_test)), "ClassId": new_test_label})

    # Create sample submission
    sample_submission = answers_df.copy()
    sample_submission["ClassId"] = 0

    # Save files
    (public / "timit_11" / "timit_11").mkdir(parents=True, exist_ok=True)

    np.save(public / "timit_11" / "timit_11" / "train_11.npy", new_train)
    np.save(public / "timit_11" / "timit_11" / "train_label_11.npy", new_train_label)
    np.save(public / "timit_11" / "timit_11" / "test_11.npy", new_test)
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)
    answers_df.to_csv(private / "answers.csv", index=False)

    # Sanity checks
    assert (
        public / "timit_11" / "timit_11" / "train_11.npy"
    ).exists(), "`train_11.npy` doesn't exist!"
    assert (
        public / "timit_11" / "timit_11" / "train_label_11.npy"
    ).exists(), "`train_label_11.npy` doesn't exist!"
    assert (
        public / "timit_11" / "timit_11" / "test_11.npy"
    ).exists(), "`test_11.npy` doesn't exist!"
    assert (public / "sampleSubmission.csv").exists(), "`sampleSubmission.csv` doesn't exist!"
    assert (private / "answers.csv").exists(), "`answers.csv` doesn't exist!"

    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected {len(old_train)} samples in combined new train and test splits, got {len(new_train) + len(new_test)}!"
