from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int8",
        "C": "int8",
        "time_step": "float64",
        "u_in": "float64",
        "u_out": "int8",
        "pressure": "float64",
    }

    old_train = read_csv(raw / "train.csv", dtype=dtypes)

    # Group by 'breath_id' and maintain the groups as lists of indices
    groups = [df.index.tolist() for _, df in old_train.groupby("breath_id")]

    # Split the groups into train and test sets such that train and test sets
    # do not contain the same 'breath_id's
    train_idx, test_idx = train_test_split(groups, test_size=0.1, random_state=0)

    # Flatten the list of indices to get indices for train and test sets
    train_idx = [idx for sublist in train_idx for idx in sublist]
    test_idx = [idx for sublist in test_idx for idx in sublist]

    # Create train and test DataFrames using the indices
    new_train = old_train.loc[train_idx]
    new_test = old_train.loc[test_idx]

    # Reset the 'id' column of new_train and new_test, starting at 1
    new_train["id"] = range(1, len(new_train) + 1)
    new_test["id"] = range(1, len(new_test) + 1)

    assert set(new_train["breath_id"]).isdisjoint(
        set(new_test["breath_id"])
    ), "Test set contains breath_ids that are in the train set"

    # Create public test
    new_test_without_labels = new_test.drop(columns=["pressure"])

    # Create sample submission
    sample_submission = new_test_without_labels.copy()[["id"]]
    sample_submission["pressure"] = 0

    # Write CSVs
    new_train.to_csv(public / "train.csv", index=False, float_format="%.10g")
    new_test_without_labels.to_csv(public / "test.csv", index=False, float_format="%.10g")
    sample_submission.to_csv(public / "sample_submission.csv", index=False, float_format="%.10g")
    new_test.to_csv(private / "test.csv", index=False, float_format="%.10g")

    # Checks
    assert len(old_train) == len(new_train) + len(
        new_test
    ), "New train and test should sum up to the old train size"
    assert (
        sample_submission.shape[0] == new_test_without_labels.shape[0]
    ), "Sample submission and new_test should have the same number of rows"
    assert sample_submission.shape[1] == 2, "Sample submission should have 2 columns"
    assert (
        new_test_without_labels.shape[1] == 7
    ), f"Expected 7 columns in new_test_without_labels, but got {new_test_without_labels.shape[1]}"
    assert new_train.shape[1] == 8, f"Expected 8 columns in new_train, but got {new_train.shape[1]}"
