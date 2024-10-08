import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv
from PIL import Image
from sklearn.model_selection import train_test_split

from mlebench.utils import extract


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Create new train and test splits
    extract(raw / "train.zip", public)

    # Original test ratio has Train set - 144 samples; Test set - 72 samples (33% ratio)
    # We set new test ratio to 0.2 to keep it number of train samples at same OOM while having enough samples for new test
    old_train_img_fpaths = sorted((public / "train").glob("*.png"))
    new_train_img_fpaths, new_test_img_fpaths = train_test_split(
        old_train_img_fpaths,
        test_size=0.2,
        random_state=0,
    )

    new_test_imgs = set([fpath.name for fpath in new_test_img_fpaths])

    # Create `test` directory
    (public / "test").mkdir(exist_ok=True, parents=True)

    for fpath in (public / "train").glob("*.png"):
        if fpath.name in new_test_imgs:
            shutil.move(fpath, public / "test" / fpath.name)

    # Create public `train_cleaned` directories
    extract(raw / "train_cleaned.zip", public)

    (private / "train_cleaned").mkdir(exist_ok=True, parents=True)

    for fpath in (public / "train_cleaned").glob("*.png"):
        if fpath.name in new_test_imgs:
            shutil.move(fpath, private / "train_cleaned" / fpath.name)

    # Write `answers.csv`
    dfs = []

    for fpath in sorted((private / "train_cleaned").glob("*.png")):
        df = to_df(fpath)
        dfs.append(df)

    answers = DataFrame(pd.concat(dfs))
    answers.to_csv(private / "answers.csv", index=False)

    shutil.rmtree(private / "train_cleaned")

    # Write `sampleSubmission.csv`
    sample_submission = answers.copy()
    sample_submission["value"] = 1
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

    # Sanity checks
    assert len(list((public / "train").glob("*.png"))) == len(old_train_img_fpaths) - len(
        new_test_img_fpaths
    ), (
        f"Expected the number of images in `train` to be "
        f"{len(old_train_img_fpaths) - len(new_test_img_fpaths)}, but got "
        f"{len(list((public / 'train').glob('*.png')))}."
    )

    assert len(list((public / "test").glob("*.png"))) == len(new_test_img_fpaths), (
        f"Expected {len(new_test_img_fpaths)} in the `test` directory, but got "
        f"{len(list((public / 'test').glob('*.png')))}."
    )

    assert read_csv(private / "answers.csv").drop_duplicates().shape[0] == len(
        read_csv(private / "answers.csv")
    ), (
        f"Expected `answers.csv` to have unique rows, but got "
        f"{read_csv(private / 'answers.csv').drop_duplicates().shape[0]} unique rows and "
        f"{len(read_csv(private / 'answers.csv'))} rows in total."
    )

    assert len(read_csv(private / "answers.csv")) == len(
        read_csv(public / "sampleSubmission.csv")
    ), (
        f"Expected `answers.csv` and `sampleSubmission.csv` to have the same number of rows, but "
        f"got {len(read_csv(private / 'answers.csv'))} rows in `answers.csv` and "
        f"{len(read_csv(public / 'sampleSubmission.csv'))} rows in `sampleSubmission.csv`."
    )

    assert "id" in read_csv(private / "answers.csv").columns, (
        f"Expected `answers.csv` to have an 'id' column, but got "
        f"{read_csv(private / 'answers.csv').columns}."
    )

    assert "value" in read_csv(private / "answers.csv").columns, (
        f"Expected `answers.csv` to have a 'value' column, but got "
        f"{read_csv(private / 'answers.csv').columns}."
    )

    assert "id" in read_csv(public / "sampleSubmission.csv").columns, (
        f"Expected `sampleSubmission.csv` to have an 'id' column, but got "
        f"{read_csv(public / 'sampleSubmission.csv').columns}."
    )

    assert "value" in read_csv(public / "sampleSubmission.csv").columns, (
        f"Expected `sampleSubmission.csv` to have a 'value' column, but got "
        f"{read_csv(public / 'sampleSubmission.csv').columns}."
    )


def to_df(img: Path) -> DataFrame:
    """Converts an image to a DataFrame, where each row corresponds to a pixel."""

    image = Image.open(img).convert("L")
    image_array = np.array(image) / 255.0

    rows, cols = image_array.shape
    data = {"id": [], "value": []}

    for row in range(rows):
        for col in range(cols):
            pixel_id = f"{img.stem}_{row+1}_{col+1}"
            pixel_value = image_array[row, col]
            data["id"].append(pixel_id)
            data["value"].append(pixel_value)

    df = DataFrame(data)

    assert (
        len(df) == rows * cols
    ), f"Expected the DataFrame to have {rows * cols} rows, but got {len(df)} rows."

    return DataFrame(data)
