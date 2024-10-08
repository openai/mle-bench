import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv

# columns in train.csv
train_columns = [
    "site_id",
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "age",
    "cancer",
    "biopsy",
    "invasive",
    "BIRADS",
    "implant",
    "density",
    "machine_id",
    "difficult_negative_case",
]
# columns in test.csv
test_columns = [
    "site_id",
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "age",
    "implant",
    "machine_id",
    "prediction_id",
]

# columns in answers/submission
submission_columns = ["prediction_id", "cancer"]


DEV = False


def prepare(raw: Path, public: Path, private: Path):

    old_train = read_csv(raw / "train.csv")
    # work on 5k samples for now, instead of 54k
    if DEV:
        old_train = old_train.sample(5000, random_state=42)

    # "You can expect roughly 8,000 patients" in the test set
    # so, split on patients. There are 11913 patients in train set
    patient_ids = old_train["patient_id"].unique()
    # Original ratio is 8000/ (8000 + 11913) ~ 0.4
    # We use 0.1 to avoid taking too many samples out of train
    train_patients, test_patients = train_test_split(patient_ids, test_size=0.1, random_state=42)
    train_patients = set(train_patients)
    test_patients = set(test_patients)

    new_train = old_train[old_train["patient_id"].isin(train_patients)].copy()
    # dont index the `prediction_id` (last col) since its not in train and we need to build it
    # index also the `cancer` column, which we'll drop later for the without_labels version
    answers = old_train[old_train["patient_id"].isin(test_patients)][
        test_columns[:-1] + ["cancer"]
    ].copy()

    new_train.to_csv(public / "train.csv", index=False)

    answers["prediction_id"] = answers["patient_id"].astype(str) + "_" + answers["laterality"]
    new_test_without_labels = answers.drop(columns=["cancer"])
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # merge multiple prediction_ids for the same patient into one for sample_submission and private test
    answers = answers[submission_columns].copy()
    # just take the first label for each prediction id -- the rest will be identical duplicates
    answers = answers.groupby("prediction_id").first().reset_index()
    answers.to_csv(private / "answers.csv", index=False)

    sample_submission = answers.copy()
    sample_submission["cancer"] = new_train.cancer.mean()  # mean cancer rate in train set
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    assert len(sample_submission) == len(
        answers
    ), "sample_submission and test.csv should have the same number of rows"
    assert len(new_test_without_labels) + len(new_train) == len(
        old_train
    ), "The sum of the rows in new_test_without_labels and new_train should be equal to the number of rows in old_train"
    # because of the merging
    assert len(answers) != len(
        new_test_without_labels
    ), "new_test and new_test_without_labels should have different number of rows"

    assert (
        answers.columns.tolist() == submission_columns
    ), f"answers should have columns {submission_columns}"
    assert (
        sample_submission.columns.tolist() == submission_columns
    ), f"sample_submission should have columns {submission_columns}"

    assert (
        new_train.columns.tolist() == old_train.columns.tolist()
    ), f"new_train should have columns {old_train.columns.tolist()}, got {new_train.columns.tolist()}"
    assert (
        new_test_without_labels.columns.tolist() == test_columns
    ), f"new_test_without_labels should have columns {test_columns}, got {new_test_without_labels.columns.tolist()}"

    assert set(new_test_without_labels["patient_id"]).isdisjoint(
        set(new_train["patient_id"])
    ), "new_test_without_labels and new_train should have disjoint patient_ids"

    # finally, split the images
    (public / "train_images").mkdir(exist_ok=True)
    for patient_id in tqdm(train_patients, total=len(train_patients)):
        patient_id_str = str(patient_id)
        patient_dir = public / "train_images" / patient_id_str
        patient_dir.mkdir(exist_ok=True)
        image_ids = new_train[new_train["patient_id"] == patient_id]["image_id"].to_list()
        for image_id in image_ids:
            shutil.copy(raw / "train_images" / patient_id_str / f"{image_id}.dcm", patient_dir)

    (public / "test_images").mkdir(exist_ok=True)
    for patient_id in tqdm(test_patients, total=len(test_patients)):
        patient_id_str = str(patient_id)
        patient_dir = public / "test_images" / patient_id_str
        patient_dir.mkdir(exist_ok=True)
        image_ids = new_test_without_labels[new_test_without_labels["patient_id"] == patient_id][
            "image_id"
        ].to_list()
        for image_id in image_ids:
            shutil.copy(raw / "train_images" / patient_id_str / f"{image_id}.dcm", patient_dir)

    # final checks
    assert len(list((public / "train_images").rglob("*.dcm"))) == len(
        new_train
    ), "Number of images in train_images should be equal to the number of rows in new_train"
    assert len(list((public / "test_images").rglob("*.dcm"))) == len(
        new_test_without_labels
    ), "Number of images in test_images should be equal to the number of rows in new_test_without_labels"
