import csv
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import compress, extract, read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Extract
    extract(raw / "en_test_2.csv.zip", raw)  # We only use the 2nd stage test set
    extract(raw / "en_train.csv.zip", raw)
    extract(raw / "en_sample_submission_2.csv.zip", raw)

    # Create train and test splits from train set
    old_train = read_csv(raw / "en_train.csv")

    # We split so that we don't share any sentence_ids between train and test
    # This gives us len(new_train) = 8924976 and len(answers) = 993465
    # Original was len(old_train) = 9918441 and len(old_test) = 956046
    unique_sentence_ids = old_train["sentence_id"].unique()
    train_sentence_ids, test_sentence_ids = train_test_split(
        unique_sentence_ids, test_size=0.1, random_state=0
    )
    new_train = old_train[old_train["sentence_id"].isin(train_sentence_ids)]
    answers = old_train[old_train["sentence_id"].isin(test_sentence_ids)]
    assert set(new_train["sentence_id"]).isdisjoint(
        set(answers["sentence_id"])
    ), f"sentence_id is not disjoint between train and test sets"

    # "sentence_id" counts need to be reset for new_train and answers
    new_train_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(new_train["sentence_id"].unique())
    }
    new_train.loc[:, "sentence_id"] = new_train["sentence_id"].map(new_train_id_mapping)
    answers_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(answers["sentence_id"].unique())
    }
    answers.loc[:, "sentence_id"] = answers["sentence_id"].map(answers_id_mapping)

    # Create new test set
    new_test = answers.drop(["after", "class"], axis=1).copy()

    # Reformat answers to match sample submission format
    answers = answers[["sentence_id", "token_id", "after"]].copy()
    answers["id"] = answers["sentence_id"].astype(str) + "_" + answers["token_id"].astype(str)
    answers = answers[["id", "after"]]

    # Create sample submission
    sample_submission = new_test[["sentence_id", "token_id", "before"]].copy()
    sample_submission["id"] = (
        sample_submission["sentence_id"].astype(str)
        + "_"
        + sample_submission["token_id"].astype(str)
    )
    sample_submission["after"] = sample_submission["before"]
    sample_submission = sample_submission[["id", "after"]]

    # Checks
    assert new_train.columns.tolist() == [
        "sentence_id",
        "token_id",
        "class",
        "before",
        "after",
    ], f"new_train.columns.tolist() == {new_train.columns.tolist()}"
    assert new_test.columns.tolist() == [
        "sentence_id",
        "token_id",
        "before",
    ], f"new_test.columns.tolist() == {new_test.columns.tolist()}"
    assert sample_submission.columns.tolist() == [
        "id",
        "after",
    ], f"sample_submission.columns.tolist() == {sample_submission.columns.tolist()}"
    assert answers.columns.tolist() == [
        "id",
        "after",
    ], f"answers.columns.tolist() == {answers.columns.tolist()}"
    assert len(new_test) + len(new_train) == len(
        old_train
    ), f"New train and test sets do not sum to old train set, got {len(new_test) + len(new_train)} and {len(old_train)}"

    # Write CSVs
    answers.to_csv(
        private / "answers.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    sample_submission.to_csv(
        private / "sample_submission.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    new_train.to_csv(
        public / "en_train.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    new_test.to_csv(
        public / "en_test_2.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    sample_submission.to_csv(
        public / "en_sample_submission_2.csv",
        index=False,
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )

    # Zip up
    with zipfile.ZipFile(public / "en_train.csv.zip", "w") as zipf:
        zipf.write(public / "en_train.csv", arcname="en_train.csv")
    with zipfile.ZipFile(public / "en_test_2.csv.zip", "w") as zipf:
        zipf.write(public / "en_test_2.csv", arcname="en_test_2.csv")
    with zipfile.ZipFile(public / "en_sample_submission_2.csv.zip", "w") as zipf:
        zipf.write(public / "en_sample_submission_2.csv", arcname="en_sample_submission_2.csv")
    (public / "en_train.csv").unlink()
    (public / "en_test_2.csv").unlink()
    (public / "en_sample_submission_2.csv").unlink()
