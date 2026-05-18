import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import get_logger

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    DEV = False

    if DEV:
        batch_cutoff = 66  # 66 instead of 660 when in dev mode
    else:
        batch_cutoff = None

    logger.info("Loading raw metadata")
    old_train = pd.read_parquet(raw / "train_meta.parquet")

    # this has batch_id and event_id, we will do a test-train split based on batch_id
    # each batch id is equally sized so we can proceed with a simple split
    batch_ids = old_train["batch_id"].unique()[:batch_cutoff]

    logger.info("Splitting batches into train and test")
    train_batch_ids, test_batch_ids = train_test_split(batch_ids, test_size=0.1, random_state=0)

    # new column tracking the split
    old_train["split"] = None
    old_train.loc[old_train["batch_id"].isin(train_batch_ids), "split"] = "train"
    old_train.loc[old_train["batch_id"].isin(test_batch_ids), "split"] = "test"

    new_train = (
        old_train[old_train["split"] == "train"]
        .drop(columns=["split"])
        .reset_index(drop=True)
        .copy()
    )
    new_test = (
        old_train[old_train["split"] == "test"]
        .drop(columns=["split"])
        .reset_index(drop=True)
        .copy()
    )

    logger.info("Creating label-less test and sample submission")
    new_test_without_labels = new_test.drop(columns=["azimuth", "zenith"])

    # match sample submission format
    new_test = new_test[["event_id", "azimuth", "zenith"]]

    # copy the format as the private test and fill dummy values like kaggle.com
    sample_submission = new_test.copy()
    sample_submission["azimuth"] = 1
    sample_submission["zenith"] = 1

    logger.info("Saving files")
    # save the prepared tables
    new_train.to_parquet(public / "train_meta.parquet", index=False, engine="fastparquet")
    new_test_without_labels.to_parquet(
        public / "test_meta.parquet", index=False, engine="fastparquet"
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    logger.info("Copying remaining files")

    # sensor_geometry can be copied as is
    shutil.copy(raw / "sensor_geometry.csv", public / "sensor_geometry.csv")

    # copy the raw train files to train and test folders respectively
    train_batch_ids = set(train_batch_ids)
    train_dest = public / "train"
    train_dest.mkdir(exist_ok=True, parents=True)
    test_batch_ids = set(test_batch_ids)
    test_dest = public / "test"
    test_dest.mkdir(exist_ok=True, parents=True)
    for batch_file in tqdm(
        sorted((raw / "train").glob("*.parquet")), desc="Copying batch parquet files"
    ):
        batch_id = int(
            batch_file.stem.split("_")[-1]
        )  # i.e. go from e.g. 'train_000.parquet' to '000' to 0
        if batch_id in train_batch_ids:
            shutil.copy(batch_file, train_dest / batch_file.name)
        elif batch_id in test_batch_ids:
            shutil.copy(batch_file, test_dest / batch_file.name)

    logger.info("Running checks")
    # Asserts
    assert len(list(public.glob("train/*.parquet"))) == len(
        train_batch_ids
    ), "Not all train batches copied"
    assert len(list(public.glob("test/*.parquet"))) == len(
        test_batch_ids
    ), "Not all test batches copied"
    assert len(train_batch_ids) + len(test_batch_ids) == len(
        batch_ids
    ), "Something went wrong with splitting the batches"

    assert len(new_train) + len(new_test) == len(
        old_train[old_train["split"].notnull()]
    ), "Expected train + test to equal the original data"
    assert len(sample_submission) == len(
        new_test
    ), "Length mismatch between private test and sample submission"

    assert sample_submission.columns.equals(
        new_test.columns
    ), f"Column mismatch between sample_submission and private test"
    assert new_train.columns.equals(
        old_train.drop(columns=["split"]).columns
    ), f"Unexpected columns in train, expected {old_train.columns}, got {new_train.columns}"
    assert new_test_without_labels.columns.equals(
        old_train.drop(columns=["azimuth", "zenith", "split"]).columns
    ), f"Unexpected columns in test, expected {old_train.drop(columns=['azimuth', 'zenith']).columns}, got {new_test_without_labels.columns}"

    assert (
        len(set(new_train["event_id"]).intersection(set(new_test["event_id"]))) == 0
    ), "Event ids overlap between train and test"
    assert set(new_test["event_id"]) == set(
        sample_submission["event_id"]
    ), "Event ids mismatch between test and sample submission"
    logger.info("Done.")
