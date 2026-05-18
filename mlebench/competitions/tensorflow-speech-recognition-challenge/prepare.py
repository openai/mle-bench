import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import extract

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",
]


@dataclass(frozen=True)
class AudioFile:
    label: str
    path: Path


def prepare(raw: Path, public: Path, private: Path):
    # extract only what we need
    extract(raw / "train.7z", raw)

    # Create train, test from train split
    audio_dir = raw / "train" / "audio"
    audio_files = sorted(
        [AudioFile(fpath.parent.name, fpath) for fpath in audio_dir.rglob("*.wav")],
        key=lambda x: f"{x.label}_{x.path.name}",
    )
    train_files, test_files = train_test_split(audio_files, test_size=0.1, random_state=0)

    # Make necessary directories
    labels = list(
        dict.fromkeys([file.label for file in train_files])
    )  # Gets unique elements deterministically

    for label in labels:
        (public / "train" / "audio" / label).mkdir(parents=True, exist_ok=True)

    (public / "test" / "audio").mkdir(parents=True, exist_ok=True)

    # Copy over train and test files
    for file in tqdm(train_files, desc="Copying train files"):
        shutil.copyfile(
            src=file.path,
            dst=public / "train" / "audio" / file.label / file.path.name,
        )

    test_records = []

    for idx, file in enumerate(tqdm(test_files, desc="Copying test files")):
        # Rename files, since training audio files across labels aren't necessarily unique.
        new_id = str(idx).zfill(8)
        new_name = f"clip_{new_id}.wav"
        test_records.append({"fname": new_name, "label": file.label})

        shutil.copyfile(
            src=file.path,
            dst=public / "test" / "audio" / new_name,
        )

    test = pd.DataFrame.from_records(test_records)
    test.to_csv(private / "test.csv", index=False)

    test_without_labels = test.drop(columns=["label"])
    sample_submission = test_without_labels.copy()
    sample_submission["label"] = "silence"
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Sanity checks
    test_audio_files = list((public / "test" / "audio").glob("*.wav"))
    num_test_files = len(test_audio_files)
    num_submission_entries = len(sample_submission)
    assert num_test_files == num_submission_entries, (
        f"The number of test audio files ({num_test_files}) does not match the number of entries "
        f"in sample_submission.csv ({num_submission_entries}). Please check the file copying process."
    )
