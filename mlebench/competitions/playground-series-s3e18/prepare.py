from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["EC1"] = 0.5
    sample_submission["EC2"] = 0.5
    sample_submission.drop(
        sample_submission.columns.difference(["id", "EC1", "EC2"]), axis=1, inplace=True
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"], axis=1, inplace=True)
    new_test.to_csv(public / "test.csv", index=False)

    # Checks
    assert new_test.shape[1] == 32, "Public test set should have 32 columns"
    assert new_train.shape[1] == 38, "Public train set should have 38 columns"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
