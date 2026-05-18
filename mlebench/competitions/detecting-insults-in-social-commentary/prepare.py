import shutil
from pathlib import Path

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # the test set labels are provided so we can just copy things without doing the split ourselves
    shutil.copy(raw / "train.csv", public / "train.csv")
    shutil.copy(raw / "test_with_solutions.csv", private / "test.csv")

    # the public test currently online is for the verification stage, which we are ignoring
    # (we are ignoring because there is some train/test leakage and unclear how this factors in leaderboard)
    # we can recover the original public test set by removing the labels from the private test set
    # can make the gold submission while we're at it
    gold_submission = read_csv(private / "test.csv")
    gold_submission = gold_submission[["Insult", "Date", "Comment"]]
    gold_submission.to_csv(private / "gold_submission.csv", index=False)

    public_test = gold_submission.drop(columns=["Insult"]).copy()
    public_test.to_csv(public / "test.csv", index=False)

    # finally, we also make our own sample_submission, same reasons as public test
    # but match the format of what's online
    sample_submission = gold_submission.copy()
    sample_submission["Insult"] = 0
    sample_submission.to_csv(public / "sample_submission_null.csv", index=False)

    # checks
    public_train = read_csv(public / "train.csv")
    public_test = read_csv(public / "test.csv")
    private_test = read_csv(private / "test.csv")

    # no `Id` column in train, so we check comment content instead
    assert public_train.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
    ], "Train columns should be Insult, Date, Comment"
    assert public_test.columns.to_list() == [
        "Date",
        "Comment",
    ], "Test columns should be Date, Comment"
    assert sample_submission.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
    ], "Sample submission columns should be Insult, Date, Comment"
    assert gold_submission.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
    ], "Gold submission columns should be Insult, Date, Comment"
    assert private_test.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
        "Usage",
    ], "Private test columns should be Insult, Date, Comment, Usage"

    assert set(public_train["Comment"]).isdisjoint(
        set(public_test["Comment"])
    ), "None of the test comments should be in the train comments"
    assert public_test.equals(
        private_test.drop(columns=["Insult", "Usage"])
    ), "Public test should be identical to private test, modulo the Insult and Usage columns"
    assert set(public_test["Comment"]) == set(
        sample_submission["Comment"]
    ), "Public test and sample submission should have the same Comments"
    assert set(public_test["Comment"]) == set(
        gold_submission["Comment"]
    ), "Public test and gold submission should have the same Comments"
