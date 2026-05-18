import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    old_train = read_csv(raw / "train.csv")
    old_dataset_info = read_csv(raw / "HuBMAP-20-dataset_information.csv")

    new_train, new_test = train_test_split(old_train, train_size=12, test_size=3, random_state=0)
    # dataset info doesnt have an id column, so quickly add it so that we can filter out old test
    old_dataset_info["id"] = old_dataset_info["image_file"].str.replace(".tiff", "")
    dataset_info = old_dataset_info[old_dataset_info["id"].isin(old_train["id"])]
    # put height and width in new_test, for grading
    new_test = new_test.merge(dataset_info[["id", "width_pixels", "height_pixels"]], on="id")
    dataset_info = dataset_info.drop(columns=["id"], inplace=False)

    dataset_info.to_csv(public / "HuBMAP-20-dataset_information.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    sample_submission = new_test[["id"]].copy()
    sample_submission["predicted"] = ""
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # basically the same as new_test but with a different column name
    gold_submission = sample_submission.copy()
    gold_submission["predicted"] = new_test["encoding"]
    gold_submission.to_csv(private / "gold_submission.csv", index=False)

    (public / "train").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_train["id"], desc="Copying train images"):
        shutil.copy(raw / "train" / f"{image_id}.tiff", public / "train" / f"{image_id}.tiff")
        shutil.copy(raw / "train" / f"{image_id}.json", public / "train" / f"{image_id}.json")
        shutil.copy(
            raw / "train" / f"{image_id}-anatomical-structure.json",
            public / "train" / f"{image_id}-anatomical-structure.json",
        )

    (public / "test").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_test["id"], desc="Copying test images"):
        shutil.copy(raw / "train" / f"{image_id}.tiff", public / "test" / f"{image_id}.tiff")
        shutil.copy(raw / "train" / f"{image_id}.json", public / "test" / f"{image_id}.json")
        shutil.copy(
            raw / "train" / f"{image_id}-anatomical-structure.json",
            public / "test" / f"{image_id}-anatomical-structure.json",
        )

    # for some reason sample_submission.csv is also in test/
    shutil.copy(public / "sample_submission.csv", public / "test" / "sample_submission.csv")

    # Checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"

    assert new_train.columns.to_list() == [
        "id",
        "encoding",
    ], "Public train set should have 2 columns, called 'id' and 'encoding'"
    assert new_test.columns.to_list() == [
        "id",
        "encoding",
        "width_pixels",
        "height_pixels",
    ], "Private test set should have 2 columns called 'id' and 'encoding'"

    assert len(sample_submission) == len(new_test), "Sample submission length should match test set"
    assert sample_submission.columns.to_list() == [
        "id",
        "predicted",
    ], "Sample submissions should have two columns, 'id' and 'predicted'"

    assert len(gold_submission) == len(new_test), "Gold submission length should match test set"
    assert gold_submission.columns.to_list() == [
        "id",
        "predicted",
    ], "Gold submissions should have two columns, 'id' and 'predicted'"

    assert gold_submission["predicted"].equals(
        new_test["encoding"]
    ), "Gold submission should match private test set"

    # assert no overlap in ids between train and test
    assert set(new_train["id"]).isdisjoint(
        set(new_test["id"])
    ), "Train and test ids should not overlap"

    # check that the images are copied correctly
    assert len(list((public / "train").glob("*.tiff"))) == len(
        new_train
    ), "Missing train tiff files"
    assert len(list((public / "train").glob("*-anatomical-structure.json"))) == len(
        new_train
    ), "Missing train structure json files"
    assert (
        len(list((public / "train").glob("*.json"))) == len(new_train) * 2
    ), "Missing train json files"

    assert len(list((public / "test").glob("*.tiff"))) == len(new_test), "Missing test tiff files"
    assert len(list((public / "test").glob("*-anatomical-structure.json"))) == len(
        new_test
    ), "Missing test structure json files"
