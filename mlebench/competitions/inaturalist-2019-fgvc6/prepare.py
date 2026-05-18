import json
import random
import shutil
import tarfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import get_logger

logger = get_logger(__name__)


def add_to_tar(src: Path, out: Path):

    assert src.exists(), f"Source directory `{src}` does not exist."
    assert src.is_dir(), f"Expected a directory, but got `{src}`."

    tqdm_desc = f"Taring {src.name} to {out.name}"
    file_paths = [path for path in src.rglob("*") if path.is_file()]
    total_files = len(file_paths)

    with tarfile.open(out, "w") as tar:
        for file_path in tqdm(file_paths, desc=tqdm_desc, unit="file", total=total_files):
            tar.add(file_path, arcname=file_path.relative_to(src))


def prepare(raw: Path, public: Path, private: Path):

    dev_mode = False
    image_count = 2 if dev_mode else float("inf")  # We copy over 2 images per category for dev mode

    # Extract train_val2019.tar.gz which contains images
    train_tar_path = raw / "train_val2019.tar.gz"
    train_extract_path = raw
    if not (raw / "train_val2019").exists():
        shutil.unpack_archive(train_tar_path, train_extract_path)

    # Create train, test from train split
    json_path = raw / "train2019.json"
    with open(json_path, "r", encoding="utf-8") as f:
        old_train_metadata = json.load(f)

    # Organize data by category so that we can split per-category later
    annotation_image_metadata_by_category = {}  # We'll collect both `annotations` and `images` here
    for annotation_info, image_info in list(
        zip(old_train_metadata["annotations"], old_train_metadata["images"])
    ):
        assert (
            annotation_info["image_id"] == image_info["id"]
        ), f"Mismatching image_id in annotation and image: {annotation_info['image_id']} vs {image_info['id']}"
        category_id = annotation_info["category_id"]
        if category_id not in annotation_image_metadata_by_category:
            annotation_image_metadata_by_category[category_id] = []
        annotation_image_metadata_by_category[category_id].append(
            {
                "annotation": annotation_info,
                "image": image_info,
            }
        )

    # Split train/test
    train_sample_count = 0  # Useful for tqdm later
    train_annotation_image_metadata_by_category = {}
    test_annotation_image_metadata_by_category = {}

    for category_id, annotation_image_metadata in tqdm(
        annotation_image_metadata_by_category.items(), desc="Assigning train/test splits"
    ):
        # Create split by "category" (class)
        # Original train+val has 268,243 images, test has 35,400 images, 0.12 ratio
        test_size = 0.12
        n_samples = len(annotation_image_metadata)
        if n_samples == 1:
            # If only one sample, put it in train
            train_annotations_images = annotation_image_metadata
            test_annotations_images = []
        elif n_samples < 5:  # Minimum 5 samples to ensure at least 1 in test
            num_test_samples = max(1, int(n_samples * test_size))
            train_annotations_images = annotation_image_metadata[:-num_test_samples]
            test_annotations_images = annotation_image_metadata[-num_test_samples:]
        else:
            train_annotations_images, test_annotations_images = train_test_split(
                annotation_image_metadata, test_size=test_size, random_state=0
            )

        train_annotation_image_metadata_by_category[category_id] = train_annotations_images
        test_annotation_image_metadata_by_category[category_id] = test_annotations_images
        train_sample_count += len(train_annotations_images)

    # Create new train2019.json
    new_train_metadata = (
        old_train_metadata.copy()
    )  # Keep 'info', 'categories', 'licenses' unchanged
    new_train_metadata.update(
        {
            "annotations": [],
            "images": [],
        }
    )
    for category_id, annotation_image_metadata in tqdm(
        train_annotation_image_metadata_by_category.items(),
        desc="Creating new train2019.json",
        total=len(train_annotation_image_metadata_by_category),
    ):
        for annotation_image in annotation_image_metadata:
            new_annotation = annotation_image["annotation"].copy()
            new_train_metadata["annotations"].append(new_annotation)
            new_image = annotation_image["image"].copy()
            new_train_metadata["images"].append(new_image)

    with open(public / "train2019.json", "w") as f:
        json.dump(new_train_metadata, f, indent=4, sort_keys=True)

    # Copy over val2019.json
    shutil.copy(raw / "val2019.json", public / "val2019.json")
    logger.info(f"Copied {raw / 'val2019.json'} to {public / 'val2019.json'}")

    # Create new test2019.json
    new_to_old_file_name = {}
    new_test_metadata = old_train_metadata.copy()
    del new_test_metadata["categories"]
    new_test_metadata.update(
        {
            "annotations": [],
            "images": [],
        }
    )
    # Flatten and shuffle test set so that we don't have all the same catedgories in a row
    test_annotations_images = [
        item for sublist in test_annotation_image_metadata_by_category.values() for item in sublist
    ]
    random.Random(0).shuffle(test_annotations_images)
    for idx, annotation_image in tqdm(
        enumerate(test_annotations_images),
        desc="Creating new test2019.json",
        total=len(test_annotations_images),
    ):

        new_annotation = annotation_image["annotation"].copy()
        new_test_metadata["annotations"].append(new_annotation)

        new_image = annotation_image["image"].copy()
        old_file_name = new_image["file_name"]
        # go from e.g. "train_val2019/Plants/400/d1322d13ccd856eb4236c8b888546c79.jpg" to "test2019/d1322d13ccd856eb4236c8b888546c79.jpg"
        new_file_name = "test2019/" + old_file_name.split("/")[-1]
        # keep track of things so we know what to copy later
        new_to_old_file_name[new_file_name] = old_file_name
        new_image["file_name"] = new_file_name
        new_test_metadata["images"].append(new_image)
    with open(public / "test2019.json", "w") as f:
        # The public test data, of course, doesn't have annotations
        public_new_test = new_test_metadata.copy()
        del public_new_test["annotations"]
        assert public_new_test.keys() == {
            "images",
            "info",
            "licenses",
        }, f"Public test metadata keys should be 'images', 'info', 'licenses', but found {public_new_test.keys()}"
        json.dump(public_new_test, f, indent=4, sort_keys=True)

    (public / "train_val2019").mkdir(parents=True, exist_ok=True)
    (public / "test2019").mkdir(parents=True, exist_ok=True)

    # Save private test answers
    answers_rows = []
    for image_info, annotation_info in zip(
        new_test_metadata["images"], new_test_metadata["annotations"]
    ):
        assert (
            image_info["id"] == annotation_info["image_id"]
        ), f"Mismatching image_id in image and annotation: {image_info['id']} vs {annotation_info['image_id']}"
        answers_rows.append(
            {
                "id": image_info["id"],
                "predicted": annotation_info["category_id"],
            }
        )
    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(private / "answers.csv", index=False)

    # Create new sample submission based on answers_df
    sample_df = answers_df.copy()
    sample_df["predicted"] = [random.Random(42).randint(0, 1009) for _ in range(len(sample_df))]
    sample_df.to_csv(public / "kaggle_sample_submission.csv", index=False)

    assert len(answers_df) == len(
        new_test_metadata["images"]
    ), f"Expected {len(new_test_metadata['images'])} rows in answers, but found {len(answers_df)}"
    assert len(sample_df) == len(
        answers_df
    ), f"Expected {len(answers_df)} rows in sample submission, but found {len(sample_df)}"
    assert answers_df["id"].equals(
        sample_df["id"]
    ), "Mismatched 'id' columns between answers and sample submission"

    # Copy train images
    train_images_copied = 0
    for category_id, annotation_image_metadata in tqdm(
        train_annotation_image_metadata_by_category.items(),
        desc="Copying train images grouped by category",
    ):
        for idx, annotation_image in enumerate(annotation_image_metadata):
            if dev_mode and idx >= image_count:
                break
            old_path = raw / annotation_image["image"]["file_name"]
            new_path = public / annotation_image["image"]["file_name"]
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(old_path, new_path)
            train_images_copied += 1

    # Copy test images
    test_images_copied = 0
    for image_info in tqdm(new_test_metadata["images"], desc="Copying test images"):
        if dev_mode and test_images_copied >= image_count:
            break
        old_path = raw / new_to_old_file_name[image_info["file_name"]]
        new_path = public / image_info["file_name"]
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(old_path, new_path)
        test_images_copied += 1

    logger.info(f"Copied {train_images_copied} train images and {test_images_copied} test images")

    if not dev_mode:
        assert len(list((public / "train_val2019").glob("**/*.jpg"))) == len(
            new_train_metadata["images"]
        ), f"Mismatching number of images in train_images, got {len(list((public / 'train_val2019').glob('**/*.jpg')))}"

        assert len(new_train_metadata["annotations"]) == len(
            new_train_metadata["images"]
        ), f"Mismatching number of annotations in train_metadata, got {len(new_train_metadata['annotations'])}"

    if not dev_mode:
        assert len(list((public / "test2019").glob("**/*.jpg"))) == len(
            new_test_metadata["images"]
        ), f"Mismatching number of images in test_images, got {len(list((public / 'test2019').glob('**/*.jpg')))}"

        assert len(new_test_metadata["annotations"]) == len(
            new_test_metadata["images"]
        ), f"Mismatching number of annotations in test_metadata, got {len(new_test_metadata['annotations'])}"

        assert len(new_train_metadata["annotations"]) + len(
            new_test_metadata["annotations"]
        ) == len(old_train_metadata["annotations"]), (
            f"Expected {len(old_train_metadata['annotations'])} annotations in total, but found"
            f"{len(new_train_metadata['annotations'])} in train and {len(new_test_metadata['annotations'])} in test"
        )

    # Re-compress images
    add_to_tar(
        src=public / "test2019", out=public / "test2019.tar.gz"
    )  # Add to tar but don't actually compress with gzip to save time
    add_to_tar(src=public / "train_val2019", out=public / "train_val2019.tar.gz")
    # Remove uncompressed directories
    shutil.rmtree(public / "test2019")
    shutil.rmtree(public / "train_val2019")
