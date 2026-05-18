import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    dev_mode = False

    # Create train, test from train split

    # Load old train
    with open(raw / "iwildcam2020_train_annotations.json", "r") as file:
        old_train_json = json.load(file)
    old_train_annotations = pd.DataFrame(old_train_json["annotations"])
    old_train_images = pd.DataFrame(old_train_json["images"])
    old_train_categories = pd.DataFrame(old_train_json["categories"])
    # old_train_info = pd.DataFrame(old_train_json["info"])

    # Load old test
    with open(raw / "iwildcam2020_test_information.json", "r") as file:
        old_test_json = json.load(file)
    old_test_categories = pd.DataFrame(old_test_json["categories"])

    # Create splits based on train's images' on 'location'
    test_size = 0.22  # 62894/(217959+62894) = 0.22
    train_image_locations = old_train_images["location"].unique()
    locations_new_train, locations_new_test = train_test_split(
        train_image_locations, test_size=test_size, random_state=0
    )

    # Filter old train to new train and new test based on location
    new_train_images = old_train_images[old_train_images["location"].isin(locations_new_train)]
    new_test_images = old_train_images[old_train_images["location"].isin(locations_new_test)]

    # Adjust the split to ensure around test_size of total samples are in the new test set
    while len(new_test_images) / (len(old_train_images) + len(new_test_images)) < test_size:
        # Move some locations from train to test
        location_to_move = locations_new_train[-1]
        locations_new_train = locations_new_train[:-1]
        locations_new_test = np.append(locations_new_test, location_to_move)
        new_train_images = old_train_images[old_train_images["location"].isin(locations_new_train)]
        new_test_images = old_train_images[old_train_images["location"].isin(locations_new_test)]

    while len(new_test_images) / (len(old_train_images) + len(new_test_images)) > test_size:
        # Move some locations from test to train
        location_to_move = locations_new_test[-1]
        locations_new_test = locations_new_test[:-1]
        locations_new_train = np.append(locations_new_train, location_to_move)
        new_train_images = old_train_images[old_train_images["location"].isin(locations_new_train)]
        new_test_images = old_train_images[old_train_images["location"].isin(locations_new_test)]

    # Get the image ids for new train and new test
    new_train_ids = new_train_images["id"].unique()
    new_test_ids = new_test_images["id"].unique()

    # Filter annotations based on new train and new test image ids
    new_train_annotations = old_train_annotations[
        old_train_annotations["image_id"].isin(new_train_ids)
    ]
    new_test_annotations = old_train_annotations[
        old_train_annotations["image_id"].isin(new_test_ids)
    ]
    new_train_categories = old_train_categories.copy()
    new_test_categories = old_test_categories.copy()

    # Answers
    answer_annotations = new_test_annotations[["image_id", "category_id"]].copy()
    answer_annotations.rename(columns={"image_id": "Id", "category_id": "Category"}, inplace=True)

    # Create a sample submission file
    sample_submission = answer_annotations.copy()
    np.random.seed(0)
    sample_submission["Category"] = np.random.randint(
        0, 676, size=len(sample_submission)
    )  # Uniform between 0 and 675

    # Checks
    assert set(new_train_annotations["image_id"]).isdisjoint(
        set(new_test_images["id"])
    ), "Train should not contain annotations of test images"
    assert len(new_train_ids) + len(new_test_ids) == len(
        old_train_images["id"]
    ), "The combined length of new_train_ids and new_test_ids should equal the length of old_train_images"
    # Assert that new_train_images and new_test_images have disjoint locations
    assert set(new_train_images["location"]).isdisjoint(
        set(new_test_images["location"])
    ), "Train and test images should not share locations"

    # Reform JSON files
    new_train_json = {
        "annotations": new_train_annotations.to_dict(orient="records"),
        "images": new_train_images.to_dict(orient="records"),
        "categories": new_train_categories.to_dict(orient="records"),
        "info": old_train_json["info"],
    }

    new_test_json = {
        "images": new_test_images.to_dict(orient="records"),
        "categories": new_test_categories.to_dict(orient="records"),
        "info": old_test_json["info"],
    }

    # Checks on JSONs
    assert (
        new_train_json.keys() == old_train_json.keys()
    ), "new_train_json and old_train_json should have the same keys"
    assert (
        new_test_json.keys() == old_test_json.keys()
    ), "new_test_json and old_test_json should have the same keys"

    # Write files
    answer_annotations.to_csv(private / "answers.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    with open(public / "iwildcam2020_train_annotations.json", "w") as f:
        json.dump(new_train_json, f)
    with open(public / "iwildcam2020_test_information.json", "w") as f:
        json.dump(new_test_json, f)

    # Copy over megadetector results
    shutil.copyfile(
        raw / "iwildcam2020_megadetector_results.json",
        public / "iwildcam2020_megadetector_results.json",
    )

    # Reduce the number of images copied over to 100 for dev mode
    if dev_mode:
        new_train_ids = new_train_ids[:100]
        new_test_ids = new_test_ids[:100]

    # Copy over image files
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in tqdm(new_train_ids, desc="Copying train images", unit="file"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    for file_id in tqdm(new_test_ids, desc="Copying test images", unit="file"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(list(public.glob("train/*.jpg"))) == len(
        new_train_ids
    ), "Number of train images should be equal to the number of unique image_id in the train set"
    assert len(list(public.glob("test/*.jpg"))) == len(
        new_test_ids
    ), "Number of test images should be equal to the number of unique image_id in the test set"
