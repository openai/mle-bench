from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import extract, get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    extract(raw / "Train.zip", raw)

    # Create new train and test from original train set
    logger.info(f"Creating new train and test sets from {raw}.")

    old_train = read_csv(raw / "Train.csv", dtype={"Id": str, "Tags": str})
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["Tags"])
    sample_submission = new_test_without_labels.copy()
    sample_submission["Tags"] = "javascript c# python php java"

    # Copy over files to private and public directories
    logger.info(f"Copying files to {private} and {public}.")

    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # Sanity checks
    logger.info("Performing sanity checks.")

    assert len(new_test_without_labels) == len(
        new_test
    ), f"Expected {len(new_test)} public test samples, got {len(new_test_without_labels)}."
    assert len(old_train) == len(new_train) + len(
        new_test
    ), f"Mismatch in number of samples in new train and test split! Expected {len(old_train)} samples, got {len(new_train) + len(new_test)}."
    assert len(sample_submission) == len(
        new_test
    ), f"Expected {len(new_test)} public test samples, got {len(sample_submission)}."
