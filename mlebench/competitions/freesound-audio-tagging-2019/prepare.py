import shutil
from pathlib import Path

import pandas as pd
import requests

from mlebench.utils import get_logger

from .vocabulary import CLASS_NAMES

logger = get_logger(__name__)


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename


def prepare(raw: Path, public: Path, private: Path):
    """
    Straightforward: we have access to the post-competition released test labels, so we don't need
    to create our own split here. We just copy over the raw data provided by the competition and
    download the test labels.

    Otherwise, the only work here is to convert the test data into the right format for grading:
    The original form of `test.csv` is a DataFrame with N rows and 2 columns. The first column is
    "fname" and the second column is the labels as comma-separated strings (class names). We
    convert the test labels into a binary many-hot matrix matching the shape of the submission,
    [N rows, M + 1 columns]: The first column is "fname" and the remaining M columns are the
    predictions for each class.
    """

    # Copy over everything in the raw directory
    logger.info("Copying raw data to public directory")
    # Don't copy the metadata file if it exists
    items_to_copy = [item for item in raw.iterdir() if "FSDKaggle2019.meta" not in item.name]
    for item in items_to_copy:
        dest = public / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy(item, dest)
    assert len(list(public.iterdir())) == len(
        items_to_copy
    ), "Expected all files in raw to be copied to public"

    # Download the test labels and metadata that were released after the competition
    test_url = "https://zenodo.org/records/3612637/files/FSDKaggle2019.meta.zip?download=1"
    dest_path = raw / "FSDKaggle2019.meta.zip"
    download_file(test_url, dest_path)
    logger.info(f"Downloaded file saved as {dest_path}")
    # # Unzip
    shutil.unpack_archive(dest_path, raw)
    unzipped_path = raw / "FSDKaggle2019.meta"
    logger.info(f"Unzipped file to {unzipped_path}")

    # Read test labels
    test_post_competition = pd.read_csv(unzipped_path / "test_post_competition.csv")
    private_test = test_post_competition[test_post_competition["usage"] == "Private"]
    # Create a binary many-hot matrix
    new_test_rows = []
    for idx, row in private_test.iterrows():
        fname = row["fname"]
        labels = row["labels"].split(",")
        labels = [1 if label in labels else 0 for label in CLASS_NAMES]
        new_test_rows.append([fname] + labels)
    new_test = pd.DataFrame(new_test_rows, columns=["fname"] + CLASS_NAMES)
    new_test.to_csv(private / "test.csv", index=False)

    # Check that test and submission match
    submission = pd.read_csv(public / "sample_submission.csv")
    assert len(submission) == len(
        new_test
    ), f"Expected {len(new_test)} rows in test.csv, but got {len(submission)}"
    assert (
        submission.columns[1:].tolist() == CLASS_NAMES
    ), "Expected class names to match between test.csv and sample_submission.csv"
    assert all(
        submission.columns == new_test.columns
    ), "Expected columns to match between test.csv and sample_submission.csv"
    new_test.sort_values("fname", inplace=True)
    submission.sort_values("fname", inplace=True)
    assert (
        submission["fname"].tolist() == new_test["fname"].tolist()
    ), "Expected 'fname' to match between test.csv and sample_submission.csv"

    # Remove the downloaded metadata
    dest_path.unlink()
    shutil.rmtree(unzipped_path)
