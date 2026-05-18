import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path) -> None:
    expected_train_columns = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    expected_test_columns = ["image_id"]
    expected_answers_columns = expected_train_columns
    expected_sample_submission_columns = expected_train_columns

    old_train = read_csv(raw / "train.csv")
    new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)

    assert set(new_train.columns) == set(
        expected_train_columns
    ), f"Expected `new_train` to have columns {expected_train_columns} but got {new_train.columns}"

    assert set(answers.columns) == set(
        expected_answers_columns
    ), f"Expected `answers` to have columns {expected_answers_columns} but got {answers.columns}"

    new_train_image_ids = new_train["image_id"].unique()
    new_test_image_ids = answers["image_id"].unique()
    to_new_image_id = {
        **{old_id: f"Train_{i}" for i, old_id in enumerate(new_train_image_ids)},
        **{old_id: f"Test_{i}" for i, old_id in enumerate(new_test_image_ids)},
    }

    new_train["image_id"] = new_train["image_id"].replace(to_new_image_id)
    answers["image_id"] = answers["image_id"].replace(to_new_image_id)

    new_test = answers[["image_id"]].copy()

    assert set(new_test.columns) == set(
        expected_test_columns
    ), f"Expected `new_test` to have columns {expected_test_columns} but got {new_test.columns}"

    sample_submission = answers[["image_id"]].copy()
    sample_submission[["healthy", "multiple_diseases", "rust", "scab"]] = 0.25

    assert set(sample_submission.columns) == set(
        expected_sample_submission_columns
    ), f"Expected `sample_submission` to have columns {expected_sample_submission_columns} but got {sample_submission.columns}"

    private.mkdir(exist_ok=True, parents=True)
    public.mkdir(exist_ok=True, parents=True)
    (public / "images").mkdir(exist_ok=True)

    for old_image_id in tqdm(old_train["image_id"], desc="Copying over train & test images"):
        assert old_image_id.startswith(
            "Train_"
        ), f"Expected train image id `{old_image_id}` to start with `Train_`."

        new_image_id = to_new_image_id.get(old_image_id, old_image_id)

        assert (
            raw / "images" / f"{old_image_id}.jpg"
        ).exists(), f"Image `{old_image_id}.jpg` does not exist in `{raw / 'images'}`."

        shutil.copyfile(
            src=raw / "images" / f"{old_image_id}.jpg",
            dst=public / "images" / f"{new_image_id}.jpg",
        )

    answers.to_csv(private / "test.csv", index=False)

    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
