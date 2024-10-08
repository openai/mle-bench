import shutil
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from mlebench.competitions.utils import get_ids_from_tf_records
from mlebench.utils import get_logger

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    # split on the TFRecord files.
    # There's 33126 train samples; 16 TFRecord files with 2071 samples each
    # so we take 2 arbitrary TF record files as our test set (4142 samples is ~ 10% of data)
    DEV = False
    if DEV:
        cutoff_index = 10000
    else:
        cutoff_index = None

    old_train = pd.read_csv(raw / "train.csv")[:cutoff_index]

    test_tf_records = {"train00-2071.tfrec", "train06-2071.tfrec"}
    # parse the IDs from the test tf records
    test_ids = []
    for tfrec in tqdm(test_tf_records, desc="Splitting test ids from train TFRecords"):
        test_ids.extend(get_ids_from_tf_records(raw / "tfrecords" / tfrec))

    old_train["split"] = "train"
    old_train.loc[old_train["image_name"].isin(test_ids), "split"] = "test"

    new_train = old_train[old_train["split"] == "train"].drop(columns=["split"])
    new_test = old_train[old_train["split"] == "test"].drop(columns=["split"])
    new_test_without_labels = new_test.copy()[
        ["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge"]
    ]

    # match format of sample submission
    new_test = new_test[["image_name", "target"]]

    # sample submission
    sample_submission = new_test.copy()
    sample_submission["target"] = 0

    # save the CSVs
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # split raw train files to into appropriate prepared/public/test/train directories
    # the files themselves do not contain target metadata so we are free to move them around
    # DICOMs and JPEGs
    (public / "train").mkdir(parents=True, exist_ok=True)
    (public / "jpeg" / "train").mkdir(parents=True, exist_ok=True)
    for image_name in tqdm(new_train["image_name"], desc="Train Images", total=len(new_train)):
        dcm_file = raw / "train" / f"{image_name}.dcm"
        jpg_file = raw / "jpeg" / "train" / f"{image_name}.jpg"
        shutil.copy(dcm_file, public / "train" / f"{image_name}.dcm")
        shutil.copy(jpg_file, public / "jpeg" / "train" / f"{image_name}.jpg")
    (public / "test").mkdir(parents=True, exist_ok=True)
    (public / "jpeg" / "test").mkdir(parents=True, exist_ok=True)
    for image_name in tqdm(new_test["image_name"], desc="Test Images", total=len(new_test)):
        dcm_file = raw / "train" / f"{image_name}.dcm"
        jpg_file = raw / "jpeg" / "train" / f"{image_name}.jpg"
        shutil.copy(dcm_file, public / "test" / f"{image_name}.dcm")
        shutil.copy(jpg_file, public / "jpeg" / "test" / f"{image_name}.jpg")

    # TFRecords
    train_count = 0
    test_count = 0
    tfrecords_dest_path = public / "tfrecords"
    tfrecords_dest_path.mkdir(parents=True, exist_ok=True)
    for file in tqdm(
        sorted((raw / "tfrecords").glob("train*.tfrec")), desc="Copying TFRecord files"
    ):
        record_count = file.stem.split("-")[1]  # i.e. get 2071 from train00-2071
        if file.name in test_tf_records:
            shutil.copy(file, tfrecords_dest_path / f"test{test_count:02d}-{record_count}.tfrec")
            test_count += 1
        else:
            shutil.copy(file, tfrecords_dest_path / f"train{train_count:02d}-{record_count}.tfrec")
            train_count += 1

    logger.info("Running asserts...")
    assert len(list(public.glob("train/*.dcm"))) == len(new_train), "Train DICOM count mismatch"
    assert len(list(public.glob("test/*.dcm"))) == len(new_test), "Test DICOM count mismatch"
    assert len(list(public.glob("jpeg/train/*.jpg"))) == len(new_train), "Train JPEG count mismatch"
    assert len(list(public.glob("jpeg/test/*.jpg"))) == len(new_test), "Test JPEG count mismatch"

    assert not set(new_train["image_name"]).intersection(
        new_test["image_name"]
    ), "Train/Test overlap"

    assert len(sample_submission) == len(new_test), "Sample submission length mismatch"
    assert (
        sample_submission["image_name"]
        .sort_values()
        .reset_index(drop=True)
        .equals(new_test["image_name"].sort_values().reset_index(drop=True))
    ), "Sample submission IDs mismatch"

    # check columns
    assert new_test_without_labels.columns.tolist() == [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
    ], "new_test_without_labels columns mismatch"
    assert new_train.columns.tolist() == [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "diagnosis",
        "benign_malignant",
        "target",
    ], "new_train columns mismatch"
    assert new_test.columns.tolist() == ["image_name", "target"], "new_test columns mismatch"
    assert sample_submission.columns.tolist() == [
        "image_name",
        "target",
    ], "sample_submission columns mismatch"
