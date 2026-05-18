import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Copy train images to `public/train/{1,2}/`
    shutil.copytree(
        src=raw / "train" / "1",
        dst=public / "train" / "1",
    )

    shutil.copytree(
        src=raw / "train" / "2",
        dst=public / "train" / "2",
    )

    # Create test `inklabels_rle.csv`
    inklabels_rle = read_csv(raw / "train" / "3" / "inklabels_rle.csv")

    assert (
        len(inklabels_rle) == 1
    ), f"Expected a single row in `inklabels_rle.csv`, got {len(inklabels_rle)} rows."

    img_path = raw / "train" / "3" / "ir.png"

    assert img_path.is_file(), f"Expected image file at {img_path}, but it does not exist."

    with Image.open(img_path) as img:
        width, height = img.size

    inklabels_rle["width"] = width
    inklabels_rle["height"] = height
    inklabels_rle["Id"] = "a"

    inklabels_rle.to_csv(private / "inklabels_rle.csv", index=False)

    # Write `gold_submission.csv`
    inklabels_rle.drop(columns=["width", "height"]).to_csv(
        private / "gold_submission.csv",
        index=False,
    )

    # Copy test images to `{public,private}/test/a/`
    test_imgs = list((raw / "train" / "3").rglob("*"))

    for fpath in tqdm(test_imgs, desc="Creating test images"):
        if not fpath.is_file():
            continue

        assert fpath.suffix in [
            ".png",
            ".csv",
            ".tif",
        ], f"Expected file with extension png, csv, or tif, got `{fpath.suffix}` for file `{fpath}`"

        relative_path = fpath.relative_to(raw / "train" / "3")

        if fpath.name in ["inklabels.png", "inklabels_rle.csv", "ir.png"]:
            continue  # skip test images and labels

        dst = public / "test" / "a" / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fpath, dst)  # everything else to `public`

    sample_submission = pd.DataFrame({"Id": ["a"], "Predicted": ["1 1 5 1"]})
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Sanity checks
    assert len(sample_submission) == len(inklabels_rle), (
        f"Expected {len(inklabels_rle)} rows in `sample_submission.csv`, got "
        f"{len(sample_submission)} rows."
    )

    actual_sample_submission = read_csv(public / "sample_submission.csv")
    actual_inklabels_rle = read_csv(private / "inklabels_rle.csv")

    assert (
        "Id" in actual_sample_submission.columns
    ), f"Expected column `Id` in `sample_submission.csv`."
    assert (
        "Predicted" in actual_sample_submission.columns
    ), f"Expected column `Predicted` in `sample_submission.csv`."

    assert "Id" in actual_inklabels_rle.columns, f"Expected column `Id` in `inklabels_rle.csv`."
    assert (
        "Predicted" in actual_inklabels_rle.columns
    ), f"Expected column `Predicted` in `inklabels_rle.csv`."
    assert (
        "width" in actual_inklabels_rle.columns
    ), f"Expected column `width` in `inklabels_rle.csv`."
    assert (
        "height" in actual_inklabels_rle.columns
    ), f"Expected column `height` in `inklabels_rle.csv`."

    assert len(list((public / "train" / "1").rglob("*"))) == len(
        list((raw / "train" / "1").rglob("*"))
    ), (
        f"Expected {len(list(raw / 'train' / '1').rglob('*'))} files in `public/train/1`, got "
        f"{len(list(public / 'train' / '1').rglob('*'))} files."
    )

    assert len(list((public / "train" / "2").rglob("*"))) == len(
        list((raw / "train" / "2").rglob("*"))
    ), (
        f"Expected {len(list(raw / 'train' / '2').rglob('*'))} files in `public/train/2`, got "
        f"{len(list(public / 'train' / '2').rglob('*'))} files."
    )

    n_test_actual = len(list((public / "test" / "a").rglob("*")))
    n_test_expected = len(list((raw / "train" / "3").rglob("*"))) - len(
        ["inklabels.png", "inklabels_rle.csv", "ir.png"]
    )

    assert n_test_actual == n_test_expected, (
        f"Expected " f"{n_test_expected} " f"files in `public/test/a`, got {n_test_actual} files."
    )
