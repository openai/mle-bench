from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["selected_text"])

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    assert len(new_test_without_labels) == len(
        new_test
    ), f"Expected new_test_without_labels ({len(new_test_without_labels)}) == new_test ({len(new_test)})"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected new_train ({len(new_train)}) + new_test ({len(new_test)}) == old_train ({len(old_train)})"

    # Create a sample submission file
    submission_df = new_test.copy()[["textID", "selected_text"]]
    submission_df["selected_text"] = ""

    submission_df.to_csv(public / "sample_submission.csv", index=False)
    assert len(submission_df) == len(
        new_test
    ), f"Expected submission_df ({len(submission_df)}) == new_test ({len(new_test)})"
