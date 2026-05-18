import shutil
from pathlib import Path

from pandas import read_csv
from sklearn.model_selection import train_test_split

from mlebench.utils import extract


def prepare(raw: Path, public: Path, private: Path):

    # Extract
    extract(raw / "train.tsv.zip", raw)

    # Create train and test splits from train set
    test_ratio = 0.3  # 66293/(156061+66293) = 0.3
    old_train = read_csv(raw / "train.tsv", sep="\t")
    new_train, answers = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Create public test
    new_test = answers.copy()
    new_test = new_test.drop("Sentiment", axis="columns")

    # Create sample submission
    sample_submission = answers[["PhraseId", "Sentiment"]].copy()
    sample_submission["Sentiment"] = 2

    # Checks
    assert new_train["PhraseId"].is_unique, "PhraseId in new_train should be unique"
    assert new_test["PhraseId"].is_unique, "PhraseId in new_test should be unique"
    assert set(new_train["PhraseId"]).isdisjoint(
        set(new_test["PhraseId"])
    ), "PhraseId in new_train and new_test should be disjoint"
    assert (
        new_train.shape[0] + new_test.shape[0] == old_train.shape[0]
    ), "New train and new test should have the same number of rows as the old train set"
    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), "New train and old train should have the same columns"
    assert new_test.columns.tolist() == [
        "PhraseId",
        "SentenceId",
        "Phrase",
    ], "new_test should have columns ['PhraseId', 'SentenceId', 'Phrase']"

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train.tsv", index=False, sep="\t")
    new_test.to_csv(public / "test.tsv", index=False, sep="\t")
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

    # Zip files
    shutil.make_archive(str(public / "train.tsv"), "zip", public, "train.tsv")
    shutil.make_archive(str(public / "test.tsv"), "zip", public, "test.tsv")

    # Delete unzipped files
    (public / "train.tsv").unlink()
    (public / "test.tsv").unlink()
