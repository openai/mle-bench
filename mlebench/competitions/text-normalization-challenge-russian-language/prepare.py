import csv
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import compress, extract, read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Extract
    extract(raw / "ru_test_2.csv.zip", raw)  # We only use the 2nd stage test set
    extract(raw / "ru_train.csv.zip", raw)
    extract(raw / "ru_sample_submission_2.csv.zip", raw)

    # Create train and test splits from train set
    old_train = read_csv(raw / "ru_train.csv")
    # We split so that we don't share any sentence_ids between train and test
    # This gives us len(new_train) = 9515325 and len(answers) = 1059191
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
    new_train["sentence_id"] = new_train["sentence_id"].map(new_train_id_mapping)
    answers_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(answers["sentence_id"].unique())
    }
    answers["sentence_id"] = answers["sentence_id"].map(answers_id_mapping)

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
        public / "ru_train.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    new_test.to_csv(
        public / "ru_test_2.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    sample_submission.to_csv(
        public / "ru_sample_submission_2.csv",
        index=False,
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )

    # Zip up
    with zipfile.ZipFile(public / "ru_train.csv.zip", "w") as zipf:
        zipf.write(public / "ru_train.csv", arcname="ru_train.csv")
    with zipfile.ZipFile(public / "ru_test_2.csv.zip", "w") as zipf:
        zipf.write(public / "ru_test_2.csv", arcname="ru_test_2.csv")
    with zipfile.ZipFile(public / "ru_sample_submission_2.csv.zip", "w") as zipf:
        zipf.write(public / "ru_sample_submission_2.csv", arcname="ru_sample_submission_2.csv")
    (public / "ru_train.csv").unlink()
    (public / "ru_test_2.csv").unlink()
    (public / "ru_sample_submission_2.csv").unlink()
