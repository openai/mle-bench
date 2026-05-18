import glob
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.utils import extract, read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Extract only what we need
    extract(raw / "train.zip", raw / "train")
    extract(raw / "train.csv.zip", raw / "train.csv")
    extract(raw / "test.zip", raw / "test")
    extract(raw / "test.csv.zip", raw / "test.csv")

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv/train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Make ids go 1, 2, ... for both train and test. Keep old ids so we can map ids of other files
    old_train_id_to_new = {
        old_id: new_id for new_id, old_id in enumerate(new_train["id"], start=1)
    }  # id starts from 1
    new_train["id"] = new_train["id"].map(old_train_id_to_new)

    old_test_id_to_new = {
        old_id: new_id for new_id, old_id in enumerate(new_test["id"], start=1)
    }  # id starts from 1
    new_test["id"] = new_test["id"].map(old_test_id_to_new)

    new_test_without_labels = new_test.drop(
        columns=["formation_energy_ev_natom", "bandgap_energy_ev"]
    )

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    train_paths = sorted(glob.glob(str(raw / "train/train/**/*.xyz")))
    for src in train_paths:
        id = int(Path(src).parts[-2])
        if id not in old_train_id_to_new.keys():  # Filter for train ids
            continue

        new_id = old_train_id_to_new[id]
        (public / "train" / str(new_id)).mkdir(parents=True, exist_ok=True)
        shutil.copy(src=src, dst=public / "train" / str(new_id) / "geometry.xyz")
    assert len(list(public.glob("train/**/*.xyz"))) == len(
        new_train
    ), f"Expected {len(new_train)} train geometry files, found {len(list(public.glob('train/**/*.xyz')))}"

    for src in train_paths:
        id = int(Path(src).parts[-2])
        if id not in old_test_id_to_new.keys():  # Filter for test ids
            continue

        new_id = old_test_id_to_new[id]
        (public / "test" / str(new_id)).mkdir(parents=True, exist_ok=True)
        shutil.copy(src=src, dst=public / "test" / str(new_id) / "geometry.xyz")
    assert len(list(public.glob("test/**/*.xyz"))) == len(
        new_test
    ), f"Expected {len(new_test)} test geometry files, found {len(list(public.glob('test/**/*.xyz')))}"

    # Create mock submission
    sample_submission = pd.DataFrame(
        {"id": new_test["id"], "formation_energy_ev_natom": 0.1779, "bandgap_energy_ev": 1.8892}
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    assert len(sample_submission) == len(
        new_test
    ), "Sample submission should have the same number of rows as the test set"
