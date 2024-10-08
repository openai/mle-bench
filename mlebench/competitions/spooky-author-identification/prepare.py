from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import extract, read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    target_col = "author"
    id_col = "id"

    # extract only what we need
    extract(raw / "train.zip", raw)

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=[target_col])

    # private test matches the format of sample submission
    one_hot_new_test = df_to_one_hot(
        new_test.drop(columns=["text"]),
        id_column=id_col,
        target_column=target_col,
        classes=CLASSES,
    )
    # fill the sample submission with arbitrary values (matching kaggle.com)
    sample_submission = one_hot_new_test.copy()
    sample_submission["EAP"] = 0.403493538995863
    sample_submission["HPL"] = 0.287808366106543
    sample_submission["MWS"] = 0.308698094897594

    # save files
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    one_hot_new_test.to_csv(private / "test.csv", index=False)
