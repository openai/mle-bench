from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["score"])

    # Save new train and test
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    assert len(new_train) + len(new_test) == len(old_train)
    assert len(new_test) == len(new_test_without_labels)

    # Create a sample submission file
    submission_df = new_test.copy()[["id", "score"]]
    submission_df["score"] = 0
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    assert len(submission_df) == len(new_test)
