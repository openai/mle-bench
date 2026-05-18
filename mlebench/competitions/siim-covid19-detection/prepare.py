import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    There are two tasks:
    - Image level: Object detection problem - detect the presence of pneumonia in the image using bounding boxes
    - Study level: Classification problem - classify the study into one of the four classes

    Images in train/ and test/ are stored in paths with the form {study}/{series}/{image}.

    Original train has 6,334 samples, and test "is of roughly the same scale as the training dataset".
    We'll split the original train into a new train/test split with 90/10 ratio.

    The split happens at the study level, with image level following accordingly.
    """
    DEV_MODE = False

    # Create new train_study_level.csv
    train_study = read_csv(raw / "train_study_level.csv")
    if DEV_MODE:
        # randomly sample 200 rows for development
        train_study = train_study.sample(n=200, random_state=0)
    new_train_study, new_test_study = train_test_split(train_study, test_size=0.1, random_state=0)
    new_train_study = new_train_study.sort_values(by="id")
    new_test_study = new_test_study.sort_values(by="id")
    new_train_study.to_csv(public / "train_study_level.csv", index=False)

    # Create new train_image_level.csv
    train_image = read_csv(raw / "train_image_level.csv")
    new_train_image = train_image[
        (train_image["StudyInstanceUID"] + "_study").isin(new_train_study["id"])
    ]
    new_test_image = train_image[
        (train_image["StudyInstanceUID"] + "_study").isin(new_test_study["id"])
    ]
    new_train_image = new_train_image.sort_values(by="id")
    new_test_image = new_test_image.sort_values(by="id")
    if not DEV_MODE:
        assert len(new_train_image) + len(new_test_image) == len(
            train_image
        ), f"Expected {len(train_image)} images"
    new_train_image.to_csv(public / "train_image_level.csv", index=False)

    # Copy data with shutil
    for study_id in tqdm(new_train_study["id"], desc="Copying train data"):
        study_id = study_id.replace("_study", "")
        shutil.copytree(raw / "train" / study_id, public / "train" / study_id)
    for study_id in tqdm(new_test_study["id"], desc="Copying test data"):
        study_id = study_id.replace("_study", "")
        shutil.copytree(raw / "train" / study_id, public / "test" / study_id)
    assert len(list(public.glob("train/*"))) == len(
        new_train_study
    ), f"Expected {len(new_train_study)} studies"
    assert len(list(public.glob("test/*"))) == len(
        new_test_study
    ), f"Expected {len(new_test_study)} studies"

    # Create gold answer submission
    rows = []

    """ 
    # new_test_study currently looks like:
    id,Negative for Pneumonia,Typical Appearance,Indeterminate Appearance,Atypical Appearance
    00086460a852_study,0,1,0,0
    000c9c05fd14_study,0,0,0,1
    # but for the submission we need to convert it to the following, where label is one of "negative", "typical", "indeterminate", "atypical"
    id,PredictionString
    00188a671292_study,{label} 1 0 0 1 1
    004bd59708be_study,{label} 1 0 0 1 1
    """
    for idx, row in new_test_study.iterrows():
        label = ["negative", "typical", "indeterminate", "atypical"][row[1:].argmax()]
        # Study-level task is just a classification task, so set bounding boxes all the same (1 0 0 1 1)
        # then the metric will only care about the label
        # https://www.kaggle.com/competitions/siim-covid19-detection/data
        rows.append({"id": row["id"], "PredictionString": f"{label} 1 0 0 1 1"})

    # new_test_image currently looks like this, and we only want the "label" column as the PredictionString
    """
    id,boxes,label,StudyInstanceUID
    000a312787f2_image,"[{'x': 789.28836, 'y': 582.43035, 'width': 1026.65662, 'height': 1917.30292}, {'x': 2245.91208, 'y': 591.20528, 'width': 1094.66162, 'height': 1761.54944}]",opacity 1 789.28836 582.43035 1815.94498 2499.73327 opacity 1 2245.91208 591.20528 3340.5737 2352.75472,5776db0cec75
    000c3a3f293f_image,,none 1 0 0 1 1,ff0879eb20ed
    0012ff7358bc_image,"[{'x': 677.42216, 'y': 197.97662, 'width': 867.79767, 'height': 999.78214}, {'x': 1792.69064, 'y': 402.5525, 'width': 617.02734, 'height': 1204.358}]",opacity 1 677.42216 197.97662 1545.21983 1197.75876 opacity 1 1792.69064 402.5525 2409.71798 1606.9105,9d514ce429a7
    """
    for idx, row in new_test_image.iterrows():
        rows.append({"id": row["id"], "PredictionString": row["label"]})

    answers = pd.DataFrame(rows)
    assert len(answers) == len(new_test_study) + len(
        new_test_image
    ), f"Expected {len(new_test_study) + len(new_test_image)} answers"
    answers.to_csv(private / "test.csv", index=False)

    # Create sample submission
    """ Sample submission should look like:
    id,PredictionString
    00188a671292_study,negative 1 0 0 1 1
    004bd59708be_study,negative 1 0 0 1 1
    00508faccd39_study,negative 1 0 0 1 1
    ...
    f77d7d1aebab_image,none 1 0 0 1 1
    ccc5b63ca96d_image,none 1 0 0 1 1
    5e8ac1fe2b82_image,none 1 0 0 1 1
    """
    rows = []
    for idx, row in new_test_study.iterrows():
        rows.append({"id": row["id"], "PredictionString": "negative 1 0 0 1 1"})
    for idx, row in new_test_image.iterrows():
        rows.append({"id": row["id"], "PredictionString": "none 1 0 0 1 1"})

    sample_submission = pd.DataFrame(rows)
    assert len(sample_submission) == len(new_test_study) + len(
        new_test_image
    ), f"Expected {len(new_test_study) + len(new_test_image)} answers"
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
