import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import read_csv

from .dogs import DOGS_LIST


def to_one_hot(df):
    return df_to_one_hot(df, id_column="id", target_column="breed", classes=DOGS_LIST)


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "labels.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    # one hot the private test because we will one-hot the submission, as per kaggle.com
    new_test = to_one_hot(new_test)

    # Copy over files
    new_train.to_csv(public / "labels.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    (public / "train").mkdir(exist_ok=True)

    for file_id in new_train["id"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    (public / "test").mkdir(exist_ok=True)

    for file_id in new_test["id"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(list(public.glob("train/*.jpg"))) == len(new_train)
    assert len(list(public.glob("test/*.jpg"))) == len(new_test)

    # Create a sample submission file
    submission_df = new_test.copy()
    for col in submission_df.columns[1:]:
        submission_df[col] = submission_df[col].astype("float64")
    submission_df.iloc[:, 1:] = 1 / 120
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    assert submission_df.shape == (len(new_test), 121)  # 1 id column + 120 breeds
