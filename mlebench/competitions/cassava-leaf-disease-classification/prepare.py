import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.competitions.utils import get_ids_from_tf_records
from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # need to split based on the TFRecord files, since not mentioned in the CSVs
    tfrecord_files = [
        path
        for path in sorted((raw / "train_tfrecords").iterdir())
        if path.is_file() and path.suffix == ".tfrec"
    ]

    # In the original there are 21397 train samples and they say test has ~15000 test samples, which is ~ 0.4/0.6 test/train split
    # We use 0.1 ratio to avoid removing too many samples from train
    new_train_tfrecords, new_test_tfrecords = train_test_split(
        tfrecord_files, test_size=0.1, random_state=0
    )

    # parse the IDs from the test tf records
    test_ids = []
    for path in new_test_tfrecords:
        test_ids.extend(get_ids_from_tf_records(path))

    old_train = read_csv(raw / "train.csv")

    old_train["split"] = "train"
    old_train.loc[old_train["image_id"].isin(test_ids), "split"] = "test"

    new_train = old_train[old_train["split"] == "train"].drop(columns=["split"])
    new_test = old_train[old_train["split"] == "test"].drop(columns=["split"])

    sample_submission = new_test.copy()
    sample_submission["label"] = 4

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    (public / "train_tfrecords").mkdir(parents=True, exist_ok=True)
    for i, path in tqdm(
        enumerate(new_train_tfrecords),
        desc="Copying Train TFRecords",
        total=len(new_train_tfrecords),
    ):
        length = path.stem.split("-")[1]
        new_name = f"ld_train{i:02d}-{length}.tfrec"

        shutil.copy(path, public / "train_tfrecords" / new_name)

    (public / "test_tfrecords").mkdir(parents=True, exist_ok=True)
    for i, path in tqdm(
        enumerate(new_test_tfrecords), desc="Copying Test TFRecords", total=len(new_test_tfrecords)
    ):
        length = path.stem.split("-")[1]
        new_name = f"ld_test{i:02d}-{length}.tfrec"

        shutil.copy(path, public / "test_tfrecords" / new_name)

    (public / "train_images").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_train["image_id"], desc="Copying Train Images", total=len(new_train)):
        shutil.copy(raw / "train_images" / image_id, public / "train_images")

    (public / "test_images").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_test["image_id"], desc="Copying Test Images", total=len(new_test)):
        shutil.copy(raw / "train_images" / image_id, public / "test_images")

    shutil.copy(raw / "label_num_to_disease_map.json", public / "label_num_to_disease_map.json")

    # checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Expected new train and new test lengths to sum to old train length"
    assert len(sample_submission) == len(
        new_test
    ), "Expected sample submission length to be equal to new test length"

    assert len(new_train) == sum(
        1 for _ in (public / "train_images").iterdir()
    ), "Mismatch in number of expected train images copied"
    assert len(new_test) == sum(
        1 for _ in (public / "test_images").iterdir()
    ), "Mismatch in number of expected test images copied"

    assert len(new_train_tfrecords) == sum(
        1 for _ in (public / "train_tfrecords").iterdir()
    ), "Mismatch in number of expected train TFRecords copied"
    assert len(new_test_tfrecords) == sum(
        1 for _ in (public / "test_tfrecords").iterdir()
    ), "Mismatch in number of expected test TFRecords copied"

    assert new_train.columns.tolist() == [
        "image_id",
        "label",
    ], "Expected new train columns to be ['image_id', 'label']"
    assert new_test.columns.tolist() == [
        "image_id",
        "label",
    ], "Expected new test columns to be ['image_id', 'label']"
    assert sample_submission.columns.tolist() == [
        "image_id",
        "label",
    ], "Expected sample submission columns to be ['image_id', 'label']"

    assert set(new_train["image_id"]).isdisjoint(
        new_test["image_id"]
    ), "Expected train and test image IDs to be disjoint"
