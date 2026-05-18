from pathlib import Path

from mlebench.utils import extract, read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    # extract only what is needed
    extract(raw / "train.csv.zip", raw)
    extract(raw / "test.csv.zip", raw)
    extract(raw / "test_labels.csv.zip", raw)
    extract(raw / "sample_submission.csv.zip", raw)

    # the test set is provided, so we dont have to split the train set nor form the sample submission
    train_with_labels = read_csv(raw / "train.csv")
    test_without_labels = read_csv(raw / "test.csv")
    answers = read_csv(raw / "test_labels.csv")
    sample_submission = read_csv(raw / "sample_submission.csv")
    sample_submission[CLASSES] = 0.5

    # save to public
    train_with_labels.to_csv(public / "train.csv", index=False)
    test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # save to private
    answers.to_csv(private / "test.csv", index=False)

    assert len(answers) == len(
        sample_submission
    ), "Private test set and sample submission should be of the same length"

    assert sorted(answers["id"]) == sorted(
        test_without_labels["id"]
    ), "Private and Public test IDs should match"
    assert sorted(sample_submission["id"]) == sorted(
        test_without_labels["id"]
    ), "Public test and sample submission IDs should match"
    assert (
        len(set(train_with_labels["id"]) & set(test_without_labels["id"])) == 0
    ), "Train and test IDs should not overlap"
