import os
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

from mlebench.utils import extract, read_csv


def filter_and_write_file(src: Path, dst: Path, old_id_to_new: Dict[int, int]):
    """
    Given txt file that has column 0 as rec_id, filters out rec_ids that are not in old_id_to_new and writes to dst
    """
    history_of_segments = open(src).read().splitlines()
    history_of_segments = history_of_segments[1:]
    history_of_segments = [
        (int(i.split(",")[0]), ",".join(i.split(",")[1:])) for i in history_of_segments
    ]
    history_of_segments = [
        (old_id_to_new[i[0]], i[1]) for i in history_of_segments if i[0] in old_id_to_new.keys()
    ]
    with open(dst, "w") as f:
        f.write("rec_id,[histogram of segment features]\n")
        for rec_id, labels in history_of_segments:
            f.write(f"{rec_id},{labels}\n")


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # extract only what we need
    extract(raw / "mlsp_contest_dataset.zip", raw)

    (public / "essential_data").mkdir(exist_ok=True)
    (public / "supplemental_data").mkdir(exist_ok=True)

    # Create train, test from train split
    cv_folds = read_csv(raw / "mlsp_contest_dataset/essential_data/CVfolds_2.txt")
    cv_folds = cv_folds[cv_folds["fold"] == 0].reset_index(drop=True)
    cv_folds.loc[cv_folds.sample(frac=0.2, random_state=0).index, "fold"] = 1

    old_id_to_new = {old_id: new_id for new_id, old_id in enumerate(cv_folds["rec_id"].values)}
    cv_folds["rec_id"] = cv_folds.index
    cv_folds.to_csv(public / "essential_data/CVfolds_2.txt", index=False)

    test_rec_ids = cv_folds[cv_folds["fold"] == 1]["rec_id"].values
    assert len(test_rec_ids) == 64, f"Expected 64 test rec_ids, got {len(test_rec_ids)}"

    # Update id2filename with new split
    rec_id2filename = read_csv(raw / "mlsp_contest_dataset/essential_data/rec_id2filename.txt")
    rec_id2filename = rec_id2filename[rec_id2filename["rec_id"].isin(old_id_to_new.keys())]
    rec_id2filename["rec_id"] = rec_id2filename["rec_id"].map(old_id_to_new)
    rec_id2filename.to_csv(public / "essential_data/rec_id2filename.txt", index=False)
    assert len(rec_id2filename) == len(
        cv_folds
    ), f"Expected {len(cv_folds)} entires in rec_id2filename, got {len(rec_id2filename)}"

    # Update labels with new split
    rec_labels = (
        open(raw / "mlsp_contest_dataset/essential_data/rec_labels_test_hidden.txt")
        .read()
        .splitlines()
    )
    rec_labels = rec_labels[1:]  # Ignore header line
    rec_labels_split = []
    for i in rec_labels:
        rec_id = i.split(",")[0]
        labels = ",".join(i.split(",")[1:]) if len(i.split(",")) > 1 else ""
        rec_labels_split.append((int(rec_id), labels))
    rec_labels_split = [i for i in rec_labels_split if i[0] in old_id_to_new.keys()]
    rec_labels_split = [(old_id_to_new[i[0]], i[1]) for i in rec_labels_split]

    # Public labels
    with open(public / "essential_data/rec_labels_test_hidden.txt", "w") as f:
        f.write("rec_id,[labels]\n")
        for rec_id, labels in rec_labels_split:
            if rec_id in test_rec_ids:
                labels = "?"
            if labels == "":  # Write without comma
                f.write(f"{rec_id}{labels}\n")
            else:
                f.write(f"{rec_id},{labels}\n")

    # Private labels. Create csv, with each row containing the label for a (rec_id, species_id) pair
    data = {"Id": [], "Probability": []}
    for rec_id, labels in rec_labels_split:
        if rec_id not in test_rec_ids:
            continue
        species_ids = [int(i) for i in labels.split(",") if i != ""]
        for species_id in range(0, 19):
            data["Id"].append(rec_id * 100 + species_id)
            data["Probability"].append(int(species_id in species_ids))

    pd.DataFrame(data).to_csv(private / "answers.csv", index=False)
    assert (
        len(pd.DataFrame(data)) == len(test_rec_ids) * 19
    ), f"Expected {len(test_rec_ids)*19} entires in answers.csv, got {len(pd.DataFrame(data))}"

    # Create new sample submission, following new submission format
    # http://www.kaggle.com/c/mlsp-2013-birds/forums/t/4961/new-submission-parser
    data = {
        "Id": [rec_id * 100 + species_id for rec_id in test_rec_ids for species_id in range(0, 19)],
        "Probability": 0,
    }
    pd.DataFrame(data).to_csv(public / "sample_submission.csv", index=False)
    assert (
        len(pd.DataFrame(data)) == len(test_rec_ids) * 19
    ), f"Expected {len(test_rec_ids)*19} entires in sample_submission.csv, got {len(pd.DataFrame(data))}"

    # Copy over species list
    shutil.copyfile(
        src=raw / "mlsp_contest_dataset/essential_data/species_list.txt",
        dst=public / "essential_data/species_list.txt",
    )

    # Copy over all src waves from train+test set
    (public / "essential_data/src_wavs").mkdir(exist_ok=True)
    for filename in rec_id2filename["filename"]:
        shutil.copyfile(
            src=raw / "mlsp_contest_dataset/essential_data/src_wavs" / f"{filename}.wav",
            dst=public / "essential_data/src_wavs" / f"{filename}.wav",
        )

    # Copy over train+test filtered spectrograms, segmentation examples, spectrograms, and supervised segmentation
    (public / "supplemental_data/filtered_spectrograms").mkdir(exist_ok=True)
    (public / "supplemental_data/segmentation_examples").mkdir(exist_ok=True)
    (public / "supplemental_data/spectrograms").mkdir(exist_ok=True)
    (public / "supplemental_data/supervised_segmentation").mkdir(exist_ok=True)
    for filename in rec_id2filename["filename"]:
        shutil.copyfile(
            src=raw
            / "mlsp_contest_dataset/supplemental_data/filtered_spectrograms"
            / f"{filename}.bmp",
            dst=public / "supplemental_data/filtered_spectrograms" / f"{filename}.bmp",
        )
        if os.path.exists(
            raw / "mlsp_contest_dataset/supplemental_data/segmentation_examples" / f"{filename}.bmp"
        ):
            shutil.copyfile(
                src=raw
                / "mlsp_contest_dataset/supplemental_data/segmentation_examples"
                / f"{filename}.bmp",
                dst=public / "supplemental_data/segmentation_examples" / f"{filename}.bmp",
            )
        shutil.copyfile(
            src=raw / "mlsp_contest_dataset/supplemental_data/spectrograms" / f"{filename}.bmp",
            dst=public / "supplemental_data/spectrograms" / f"{filename}.bmp",
        )
        shutil.copyfile(
            src=raw
            / "mlsp_contest_dataset/supplemental_data/supervised_segmentation"
            / f"{filename}.bmp",
            dst=public / "supplemental_data/supervised_segmentation" / f"{filename}.bmp",
        )

    # Copy over remaining files
    shutil.copyfile(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_clusters.bmp",
        dst=public / "supplemental_data/segment_clusters.bmp",
    )
    shutil.copyfile(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_mosaic.bmp",
        dst=public / "supplemental_data/segment_mosaic.bmp",
    )

    filter_and_write_file(
        src=raw / "mlsp_contest_dataset/supplemental_data/histogram_of_segments.txt",
        dst=public / "supplemental_data/histogram_of_segments.txt",
        old_id_to_new=old_id_to_new,
    )
    filter_and_write_file(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_features.txt",
        dst=public / "supplemental_data/segment_features.txt",
        old_id_to_new=old_id_to_new,
    )
    filter_and_write_file(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_rectangles.txt",
        dst=public / "supplemental_data/segment_rectangles.txt",
        old_id_to_new=old_id_to_new,
    )
