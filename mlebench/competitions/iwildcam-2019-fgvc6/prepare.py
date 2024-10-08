import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import extract, get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    dev_mode = False

    # Create train, test from train split
    # Original split ratio is 153730/(196157+153730) = 0.44
    # We use 0.1 so as to not take out too many samples from train
    test_size = 0.1
    old_train = read_csv(raw / "train.csv")
    # Create a new column 'split' and assign it randomly to 'test' or 'train' based on the value of the 'location' column
    locations = old_train["location"].unique()
    train_locations, test_locations = train_test_split(
        locations, test_size=test_size, random_state=8
    )  # We target a 44% test set size, we have empirically trialed seeds and landed on 8 to achieve this

    old_train["split"] = old_train["location"].apply(
        lambda loc: "test" if loc in test_locations else "train"
    )

    new_train = old_train[old_train["split"] == "train"].drop(columns=["split"])
    answers = old_train[old_train["split"] == "test"].drop(columns=["split"])

    logger.debug("Train locations: %s", train_locations)
    logger.debug("Test locations: %s", test_locations)
    logger.debug("Test size: %s", len(answers) / (len(new_train) + len(answers)))

    old_train = old_train.drop(columns=["split"])  # Drop helper column

    new_test = answers.copy().drop(columns=["category_id"])
    gold_submission = answers.copy()[["id", "category_id"]]
    gold_submission.rename(columns={"id": "Id", "category_id": "Category"}, inplace=True)

    # Extract only what we need
    (raw / "train_images").mkdir(exist_ok=True)
    extract(raw / "train_images.zip", raw / "train_images")
    assert len(list(raw.glob("train_images/*.jpg"))) == len(
        old_train["id"].unique()
    ), f"Raw train images should have the same number of images as the unique ids in the old train set, but got {len(list(raw.glob('train_images/*.jpg')))} files and {len(old_train['id'].unique())} ids"

    # Make sample submission
    submission_df = new_test.copy()[["id"]]
    submission_df["category_id"] = 0
    submission_df.rename(columns={"id": "Id", "category_id": "Category"}, inplace=True)

    # Checks
    assert set(new_train["id"]).isdisjoint(
        set(new_test["id"])
    ), "new_train and new_test are not disjoint"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should be equal to the length of the original train set"
    assert len(answers) == len(
        new_test
    ), "Length of answers should be equal to the length of new_test"
    assert len(submission_df) == len(
        answers
    ), "Length of answers should be equal to the length of the sample submission"
    assert (
        old_train.columns.tolist() == new_train.columns.tolist()
    ), f"new_train should have the same columns as the original train set: old_train: {old_train.columns.tolist()} != new_train: {new_train.columns.tolist()}"
    assert set(new_train["location"]).isdisjoint(
        set(new_test["location"])
    ), "new_train and new_test should not share any locations"

    # Write CSVs
    answers.to_csv(private / "test.csv", index=False)
    gold_submission.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=True)

    # Copy files
    (public / "train_images").mkdir(exist_ok=True)
    (public / "test_images").mkdir(exist_ok=True)

    if dev_mode:
        new_train = new_train.sample(n=100)
        new_test = new_test.sample(n=100)

    for file_id in tqdm(new_train["id"], desc="Copying train images"):
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}.jpg",
            dst=public / "train_images" / f"{file_id}.jpg",
        )

    for file_id in tqdm(new_test["id"], desc="Copying test images"):
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}.jpg",
            dst=public / "test_images" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(list(public.glob("test_images/*.jpg"))) == len(
        new_test["id"].unique()
    ), f"Public test images should have the same number of images as the unique ids in the test set, but got {len(list(public.glob('test_images/*.jpg')))} files and {len(new_test['id'].unique())} ids"
    assert len(list(public.glob("train_images/*.jpg"))) == len(
        new_train["id"].unique()
    ), f"Public train images should have the same number of images as the unique ids in the train set, but got {len(list(public.glob('train_images/*.jpg')))} files and {len(new_train['id'].unique())} ids"

    # Zip up image directories and delete non-zipped files
    shutil.make_archive(public / "train_images", "zip", public / "train_images")
    shutil.make_archive(public / "test_images", "zip", public / "test_images")
    shutil.rmtree(public / "train_images")
    shutil.rmtree(public / "test_images")
