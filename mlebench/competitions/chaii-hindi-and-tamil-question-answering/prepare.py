from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["answer_start", "answer_text"])

    # make private test match submission format
    new_test = new_test[["id", "answer_text"]]
    new_test.columns = ["id", "PredictionString"]

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["PredictionString"] = "dummy text"
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    assert len(sample_submission) == len(
        new_test
    ), "Sample submission length does not match test length."
