import json
import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import get_logger

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the raw data into public and private datasets with appropriate test/train splits.

    `train/metadata.json` is the "table of contents" for our data, with the following structure:
    (More details at https://www.kaggle.com/competitions/herbarium-2021-fgvc8/data)
    ```
    {
        "annotations" : [annotation],
        "categories" : [category],
        "images" : [image],
        "info" : info,
        "licenses" : [license],
        "institutions" : [region]
    }
    ```
    - `images` and `annotations` are both N-length lists corresponding to the N samples.
        We'll need to split each of these lists into train and test.
    - The other fields are dataset-wide metadata that we don't need to touch.

    - test/metadata.json is the same structure as train/metadata.json, but without "annotations", "categories", "institutions"

    Other notes:
    - train/test splits need to occur per category (each category should be in both train and test).
    - The `test/images` and `train/images` folders have nested subdirs to make it easier to browse
        - `train/images` is structured as `{category_id[:3]}/{category_id[3:]}/{image_id}.jpg`
        - `test/images` is structured as `{image_idx[:3]}/{image_idx}.jpg` (to not reveal the category)
    - When we create the new splits, we re-assign image indices so that we don't give away labels based on the index
        - train images are indexed within their own category
        - test images follow a flat index after shuffling the categories
    """

    dev_mode = False
    dev_count = 2  # Copy over n images per category when in dev mode

    # Create train, test from train split
    json_path = raw / "train/metadata.json"
    with open(json_path, "r", encoding="utf-8") as f:
        old_train_metadata = json.load(f)

    # Organize data by category so that we can split per-category later
    annotations_images_by_category = {}  # We'll collect both `annotations` and `images` here
    for annotation, image in list(
        zip(old_train_metadata["annotations"], old_train_metadata["images"])
    ):
        assert (
            annotation["image_id"] == image["id"]
        ), f"Mismatching image_id in annotation and image: {annotation['image_id']} vs {image['id']}"
        category_id = annotation["category_id"]
        if category_id not in annotations_images_by_category:
            annotations_images_by_category[category_id] = []
        annotations_images_by_category[category_id].append(
            {
                "annotation": annotation,
                "image": image,
            }
        )

    # Split train/test
    train_sample_count = 0  # Useful for tqdm later
    train_annotations_images_by_category = {}
    test_annotations_images_by_category = {}
    for category_id, annotations_images in tqdm(
        annotations_images_by_category.items(), desc="Assigning train/test splits"
    ):
        test_size = 0.2
        # Create split by "category" (class): Each category needs to be in both train and test (80:20) as per original ratio
        n_samples = len(annotations_images)
        if n_samples == 1:
            # If only one sample, put it in train
            train_annotations_images = annotations_images
            test_annotations_images = []
        elif n_samples < 5:  # Minimum 5 samples to ensure at least 1 in test
            # Ensure at least 1 sample in test
            test_size = max(1, int(n_samples * test_size))
            train_annotations_images = annotations_images[:-test_size]
            test_annotations_images = annotations_images[-test_size:]
        else:
            # Original split logic
            train_annotations_images, test_annotations_images = train_test_split(
                annotations_images, test_size=test_size, random_state=0
            )

        train_annotations_images_by_category[category_id] = train_annotations_images
        test_annotations_images_by_category[category_id] = test_annotations_images
        train_sample_count += len(train_annotations_images)

    # Add to train set
    new_train_metadata = (
        old_train_metadata.copy()
    )  # Keep 'categories', 'info', 'licenses', 'institutions'
    new_train_metadata.update(
        {
            "annotations": [],
            "images": [],
        }
    )
    with tqdm(
        desc="Creating new train dataset",
        total=train_sample_count,
    ) as pbar:
        for category_id, annotations_images in train_annotations_images_by_category.items():
            # Create a nested directory from category_id, e.g. 15504 -> "155/04" or 3 -> "000/03"
            category_subdir = f"{category_id // 100:03d}/{category_id % 100:02d}"
            (public / "train/images" / category_subdir).mkdir(exist_ok=True, parents=True)
            for idx, annotation_image in enumerate(annotations_images):
                new_annotation = annotation_image["annotation"].copy()
                new_train_metadata["annotations"].append(new_annotation)

                new_image = annotation_image["image"].copy()
                new_train_metadata["images"].append(new_image)

                # Copy file from raw to public
                if (
                    not dev_mode or idx < dev_count
                ):  # if dev_mode, only copy the first dev_count images
                    src_path = raw / "train" / annotation_image["image"]["file_name"]
                    dst_path = public / "train" / annotation_image["image"]["file_name"]
                    shutil.copyfile(src=src_path, dst=dst_path)

                pbar.update(1)

    with open(public / "train/metadata.json", "w") as f:
        json.dump(new_train_metadata, f, indent=4, sort_keys=True)

    if not dev_mode:
        assert len(list((public / "train/images").glob("**/*.jpg"))) == len(
            new_train_metadata["images"]
        ), f"Mismatching number of images in train_images, got {len(list((public / 'train/images').glob('**/*.jpg')))}"
        assert len(new_train_metadata["annotations"]) == len(
            new_train_metadata["images"]
        ), f"Mismatching number of annotations in train_metadata, got {len(new_train_metadata['annotations'])}"

    # Add to test set
    new_test_metadata = old_train_metadata.copy()
    del new_test_metadata["categories"]
    del new_test_metadata["institutions"]
    new_test_metadata.update(
        {
            "annotations": [],
            "images": [],
        }
    )
    # Flatten and shuffle test set so that we don't have all the same categories in a row
    test_annotations_images = [
        item for sublist in test_annotations_images_by_category.values() for item in sublist
    ]
    random.Random(0).shuffle(test_annotations_images)
    for idx, annotation_image in tqdm(
        enumerate(test_annotations_images),
        desc="Creating new test dataset",
        total=len(test_annotations_images),
    ):

        # Make new image id, for test set this is just the index
        new_image_id = str(idx)
        # Make new filename from image id e.g. "000/0.jpg"
        new_file_name = f"images/{idx // 1000:03d}/{idx}.jpg"

        new_annotation = annotation_image["annotation"].copy()
        new_annotation["image_id"] = new_image_id
        new_test_metadata["annotations"].append(new_annotation)

        new_image = annotation_image["image"].copy()
        new_image["id"] = new_image_id
        new_image["file_name"] = new_file_name
        new_test_metadata["images"].append(new_image)

        # Copy file from raw to public
        if not dev_mode or idx < dev_count:  # if dev_mode, only copy the first dev_count images
            src_path = raw / "train" / annotation_image["image"]["file_name"]
            dst_path = public / "test" / new_file_name
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(src=src_path, dst=dst_path)

    # Save new test metadata
    with open(public / "test/metadata.json", "w") as f:
        # The public test data, of course, doesn't have annotations
        public_new_test = new_test_metadata.copy()
        del public_new_test["annotations"]
        assert public_new_test.keys() == {
            "images",
            "info",
            "licenses",
        }, f"Public test metadata keys should be 'images', 'info', 'licenses', but found {public_new_test.keys()}"
        json.dump(public_new_test, f, indent=4, sort_keys=True)

    if not dev_mode:
        assert len(list((public / "test/images").glob("**/*.jpg"))) == len(
            new_test_metadata["images"]
        ), f"Mismatching number of images in test_images, got {len(list((public / 'test/images').glob('**/*.jpg')))}"
        assert len(new_test_metadata["annotations"]) == len(
            new_test_metadata["images"]
        ), f"Mismatching number of annotations in test_metadata, got {len(new_test_metadata['annotations'])}"
        assert len(new_train_metadata["annotations"]) + len(
            new_test_metadata["annotations"]
        ) == len(old_train_metadata["annotations"]), (
            f"Expected {len(old_train_metadata['annotations'])} annotations in total, but found"
            f"{len(new_train_metadata['annotations'])} in train and {len(new_test_metadata['annotations'])} in test"
        )

    # Save private test answers
    answers_rows = []
    for image, annotation in zip(new_test_metadata["images"], new_test_metadata["annotations"]):
        assert (
            image["id"] == annotation["image_id"]
        ), f"Mismatching image_id in image and annotation: {image['id']} vs {annotation['image_id']}"
        answers_rows.append(
            {
                "Id": image["id"],
                "Predicted": annotation["category_id"],
            }
        )
    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(private / "answers.csv", index=False)

    # Create new sample submission that matches raw/sample_submission.csv, but for the new test set
    sample_rows = []
    for image in new_test_metadata["images"]:
        sample_rows.append(
            {
                "Id": image["id"],
                "Predicted": 0,
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(public / "sample_submission.csv", index=False)

    assert len(answers_df) == len(
        new_test_metadata["images"]
    ), f"Expected {len(new_test_metadata['images'])} rows in answers, but found {len(answers_df)}"
    assert len(sample_df) == len(
        answers_df
    ), f"Expected {len(answers_df)} rows in sample submission, but found {len(sample_df)}"
    assert answers_df["Id"].equals(
        sample_df["Id"]
    ), "Mismatched 'Id' columns between answers and sample submission"
