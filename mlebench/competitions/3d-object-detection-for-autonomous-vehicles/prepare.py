import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import get_logger

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the raw train data into new train/test splits.

    There isn't detailed documentation on how the train/test split of the raw data was made, but according to
    this post https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/discussion/133895:
    - It consists of a raw camera, lidar data, and HD semantic map.
    - 180 scenes, 25s each
    - 638,000 2D and 3D annotations over 18,000 objects
    - The dataset had nine classes with a large class imbalance.
    The original train/test split:
    - Train set 40% (train.csv has 22680 rows)
    - Test set: public 30%, private 30% (sample submission has 27468 rows)

    Since there are 180 scenes and >>180 samples, each sample is not independent; samples within a scene are highly correlated
    so we can't randomly split samples at an individual level. Instead, we split by scenes.
    In practice, scenes are identified by "log tokens" in the data, corresponding to a single log file (listed in log.json).
    (We also verified that the log IDs in the raw train/test splits are disjoint, which supports this choice.)

    ## What's in the dataset?
    ```
    $ ls raw/
    sample_submission.csv   test_data/      test_images/    test_lidar/     test_maps/
    train.csv               train_data/     train_images/   train_lidar/    train_maps/

    $ ls raw/train_data/
    attribute.json          category.json  instance.json  map.json     sample_annotation.json  scene.json   visibility.json
    calibrated_sensor.json  ego_pose.json  log.json       sample.json  sample_data.json        sensor.json
    $ ls raw/train_images/ | wc -l
    158757
    $ ls raw/train_lidar | wc -l
    30744

    $ ls raw/test_data/
    attribute.json  calibrated_sensor.json  category.json  ego_pose.json  log.json  map.json  sample.json  sample_data.json  scene.json  sensor.json  visibility.json
    # test_data/ omits the instance.json and sample_annotation.json files
    $ ls raw/test_images/ | wc -l
    192276
    $ ls raw/test_lidar/ | wc -l
    27468
    """
    DEV_MODE = False
    (public / "test_data").mkdir(parents=True, exist_ok=True)
    (public / "train_data").mkdir(parents=True, exist_ok=True)

    """
    log.json
    """
    # Start the split at the log level, and the rest follows
    with open(raw / "train_data" / "log.json") as f:
        logs = json.load(f)
    log_ids = [log["token"] for log in logs]
    assert len(log_ids) == len(set(log_ids)), "Log IDs must be unique"
    logger.info(f"Found {len(log_ids)} logs")
    # Split the logs into train/test
    # previous ratio had 180 train logs and 218 test logs; we'll split the 180 train samples into 80% new train and 20% new test
    # (trying not to reduce the availibility of training data, but need a large-ish set since there are 9 object classes)
    train_log_ids, test_logs_ids = train_test_split(log_ids, test_size=0.2, random_state=0)
    logger.info(f"Train logs: {len(train_log_ids)}, Test logs: {len(test_logs_ids)}")
    with open(public / "train_data" / "log.json", "w") as f:
        json.dump([log for log in logs if log["token"] in train_log_ids], f)
    with open(public / "test_data" / "log.json", "w") as f:
        json.dump([log for log in logs if log["token"] in test_logs_ids], f)

    """
    sample.json
    """
    # Create train/test sample splits following the log split
    with open(raw / "train_data" / "sample.json") as f:
        samples = json.load(f)
    logger.info(f"Found {len(samples)} samples")
    train_samples = [sample for sample in samples if sample["scene_token"] in train_log_ids]
    test_samples = [sample for sample in samples if sample["scene_token"] in test_logs_ids]
    logger.info(f"New train samples: {len(train_samples)}, new test samples: {len(test_samples)}")
    assert len(train_samples) + len(test_samples) == len(
        samples
    ), f"New train ({len(train_samples)}) and test ({len(test_samples)}) samples must cover all samples ({len(samples)})"
    with open(public / "train_data" / "sample.json", "w") as f:
        json.dump(train_samples, f)
    with open(public / "test_data" / "sample.json", "w") as f:
        json.dump(test_samples, f)

    """
    Make train.csv
    """
    # train.csv has columns `Id` and `PredictionString`, with `PredictionString` in the following format: `center_x center_y center_z width length height yaw class_name`
    with open(raw / "train.csv") as f:
        train_df = pd.read_csv(f)
    logger.info(f"Found {len(train_df)} train rows")
    new_train_df = train_df[train_df["Id"].isin([sample["token"] for sample in train_samples])]
    new_test_df = train_df[train_df["Id"].isin([sample["token"] for sample in test_samples])]
    logger.info(f"Train rows: {len(new_train_df)}, Test rows: {len(new_test_df)}")
    assert len(new_train_df) + len(new_test_df) == len(
        train_df
    ), f"New train ({len(new_train_df)}) and test ({len(new_test_df)}) annotations must cover all annotations ({len(train_df)})"
    assert len(new_train_df) == len(
        train_samples
    ), f"New train rows ({len(new_train_df)}) must match train samples ({len(train_samples)})"
    assert len(new_test_df) == len(
        test_samples
    ), f"New test rows ({len(new_test_df)}) must match test samples ({len(test_samples)})"
    new_train_df.to_csv(public / "train.csv", index=False)

    """
    Make private test.csv
    """
    # test.csv is basically new_test_df, but the "PredictionString" column needs to have a "confidence" value added
    # so the format becomes: `confidence center_x center_y center_z width length height yaw class_name`
    def add_confidence(pred_string):
        pred_tokens = pred_string.split(" ")
        assert (
            len(pred_tokens) % 8 == 0
        ), f"Expected 8 tokens per object, but got {len(pred_tokens)}"
        new_pred_tokens = []
        for i in range(0, len(pred_tokens), 8):
            new_pred_tokens.extend(["1.0"] + pred_tokens[i : i + 8])
        return " ".join(new_pred_tokens)

    # Apply the function to the entire 'PredictionString' column
    new_test_df["PredictionString"] = new_test_df["PredictionString"].apply(add_confidence)
    new_test_df.to_csv(private / "test.csv", index=False)

    """
    Make sample_submission.csv
    """
    # sample submission is the same as test.csv but with empty prediction strings
    sample_submission = new_test_df[["Id"]].copy()
    sample_submission["PredictionString"] = ""
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    """
    Split sample_data.json
    """
    # sample_data.json is a list of all images and lidar files, and each entry has a `sample_token` field that identifies which sample it belongs to
    with open(raw / "train_data" / "sample_data.json") as f:
        sample_data = json.load(f)
    logger.info(f"Found {len(sample_data)} train sample data")
    new_train_sample_data, new_test_sample_data = [], []
    for sample_datum in sample_data:
        sample_token = sample_datum["sample_token"]
        if sample_token in new_train_df["Id"].values:
            new_train_sample_data.append(sample_datum)
        elif sample_token in new_test_df["Id"].values:
            new_test_sample_data.append(sample_datum)
        else:
            raise ValueError(
                f"Sample data token {sample_token} doesn't belong to either new train or new test set"
            )
    logger.info(
        f"New train sample data: {len(new_train_sample_data)}, new test sample data: {len(new_test_sample_data)}"
    )
    assert len(new_train_sample_data) + len(new_test_sample_data) == len(
        sample_data
    ), f"New train ({len(new_train_sample_data)}) and test ({len(new_test_sample_data)}) sample data must cover all sample data ({len(sample_data)})"
    with open(public / "train_data" / "sample_data.json", "w") as f:
        json.dump(new_train_sample_data, f)
    with open(public / "test_data" / "sample_data.json", "w") as f:
        json.dump(new_test_sample_data, f)

    """
    Copy over maps
    """
    # There is only one map which is identical in both raw train/test so no need to modify, just copy over
    # $ diff raw/test_maps/map_raster_palo_alto.png raw/train_maps/map_raster_palo_alto.png # -> no output
    (public / "test_maps").mkdir(parents=True, exist_ok=True)
    (public / "train_maps").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=raw / "train_maps" / "map_raster_palo_alto.png",
        dst=public / "test_maps" / "map_raster_palo_alto.png",
    )
    shutil.copyfile(
        src=raw / "train_maps" / "map_raster_palo_alto.png",
        dst=public / "train_maps" / "map_raster_palo_alto.png",
    )

    """
    Copy attribute.json
    """
    # attribute.json is a list of object states, there are 18 attributes in the train set including "object_action_walking", "object_action_parked", etc.
    # The raw test set has an attribute.json file drawn from the same set of attributes, but only has 17 attributes (whichever attributes
    # are present in the test set.) For simplicity, we'll just copy the full list of 18 attributes in both the new train and new test sets.
    shutil.copyfile(
        src=raw / "train_data" / "attribute.json", dst=public / "train_data" / "attribute.json"
    )
    shutil.copyfile(
        src=raw / "train_data" / "attribute.json", dst=public / "test_data" / "attribute.json"
    )

    """
    Split calibrated_sensor.json
    """
    # calibrated_sensor.json is a list of sensor calibration parameters corresponding to the setup of the sensor at the time each sample was taken.
    # This file will be split following the sample_data split (each sample_datum has a `calibrated_sensor_token`)
    with open(raw / "train_data" / "calibrated_sensor.json") as f:
        calibrated_sensors = json.load(f)
    calibration_by_calibrated_sensor_token = {cal["token"]: cal for cal in calibrated_sensors}
    new_train_calibrated_sensors, new_test_calibrated_sensors = [], []
    for sample_datum in new_train_sample_data:
        calibrated_sensor_token = sample_datum["calibrated_sensor_token"]
        if calibrated_sensor_token in [cal["token"] for cal in new_train_calibrated_sensors]:
            continue  # Each calibrated sensor is used by multiple samples, we don't need to add it multiple times
        new_train_calibrated_sensors.append(
            calibration_by_calibrated_sensor_token[calibrated_sensor_token]
        )
    for sample_datum in new_test_sample_data:
        calibrated_sensor_token = sample_datum["calibrated_sensor_token"]
        if calibrated_sensor_token in [cal["token"] for cal in new_test_calibrated_sensors]:
            continue  # Each calibrated sensor is used by multiple samples, we don't need to add it multiple times
        new_test_calibrated_sensors.append(
            calibration_by_calibrated_sensor_token[calibrated_sensor_token]
        )
    logger.info(
        f"New train calibrated sensors: {len(new_train_calibrated_sensors)}, new test calibrated sensors: {len(new_test_calibrated_sensors)}"
    )
    assert len(
        set([cal["token"] for cal in new_train_calibrated_sensors + new_test_calibrated_sensors])
    ) == len(
        calibrated_sensors
    ), f"New train and test calibrated sensors must cover all calibrated sensors ({len(calibrated_sensors)})"
    with open(public / "train_data" / "calibrated_sensor.json", "w") as f:
        json.dump(new_train_calibrated_sensors, f)
    with open(public / "test_data" / "calibrated_sensor.json", "w") as f:
        json.dump(new_test_calibrated_sensors, f)

    """
    Copy category.json
    """
    # category.json is the list of 9 object classes, and is the same for train/test
    shutil.copyfile(
        src=raw / "train_data" / "category.json", dst=public / "train_data" / "category.json"
    )
    shutil.copyfile(
        src=raw / "train_data" / "category.json", dst=public / "test_data" / "category.json"
    )

    """
    Split ego_pose.json
    """
    # ego_pose.json is a list of vehicle poses, and will be split following the sample_data split
    with open(raw / "train_data" / "ego_pose.json") as f:
        ego_poses = json.load(f)
    ego_pose_by_ego_pose_token = {ego["token"]: ego for ego in ego_poses}
    new_train_ego_poses, new_test_ego_poses = [], []
    for sample_datum in new_train_sample_data:
        ego_pose_token = sample_datum["ego_pose_token"]
        new_train_ego_poses.append(ego_pose_by_ego_pose_token[ego_pose_token])
    for sample_datum in new_test_sample_data:
        ego_pose_token = sample_datum["ego_pose_token"]
        new_test_ego_poses.append(ego_pose_by_ego_pose_token[ego_pose_token])
    logger.info(
        f"New train ego poses: {len(new_train_ego_poses)}, new test ego poses: {len(new_test_ego_poses)}"
    )
    assert len(set([ego["token"] for ego in new_train_ego_poses + new_test_ego_poses])) == len(
        ego_poses
    ), f"New train and test ego poses must cover all ego poses ({len(ego_poses)})"
    with open(public / "train_data" / "ego_pose.json", "w") as f:
        json.dump(new_train_ego_poses, f)
    with open(public / "test_data" / "ego_pose.json", "w") as f:
        json.dump(new_test_ego_poses, f)

    """
    Create map.json
    """
    # map.json is the list of maps, and a list of logs that used those maps. But in the raw dataset, we only have one map,
    # so this ends up being just a list of one map, which has a sublist of all the logs in the split.
    # [{"log_tokens": [...], "category": "semantic_prior", "filename": "maps/map_raster_palo_alto.png", "token": "53992ee3023e5494b90c316c183be829"}]
    with open(raw / "train_data" / "map.json") as f:
        maps = json.load(f)
    assert len(maps) == 1, "Expected only one map in the raw dataset"
    # Just replace the list of "log_tokens" with the new train and test log IDs
    new_train_maps = maps.copy()
    new_train_maps[0]["log_tokens"] = train_log_ids
    new_test_maps = maps.copy()
    new_test_maps[0]["log_tokens"] = test_logs_ids
    with open(public / "train_data" / "map.json", "w") as f:
        json.dump(new_train_maps, f)
    with open(public / "test_data" / "map.json", "w") as f:
        json.dump(new_test_maps, f)

    """
    Split scene.json
    """
    # scene.json is a list of scenes corresponding exactly to each log file. The scenes describe the first and last
    # samples in each scene, as well as how many samples are in each scene.
    # We'll split this following the log split.
    with open(raw / "train_data" / "scene.json") as f:
        scenes = json.load(f)
    logger.info(f"Found {len(scenes)} scenes")
    new_train_scenes, new_test_scenes = [], []
    for scene in scenes:
        log_token = scene["log_token"]
        if log_token in train_log_ids:
            new_train_scenes.append(scene)
        elif log_token in test_logs_ids:
            new_test_scenes.append(scene)
        else:
            raise ValueError(
                f"Scene log token {log_token} doesn't belong to either new train or new test set"
            )
    logger.info(
        f"New train scenes: {len(new_train_scenes)}, new test scenes: {len(new_test_scenes)}"
    )
    assert len(new_train_scenes) + len(new_test_scenes) == len(
        scenes
    ), f"New train ({len(new_train_scenes)}) and test ({len(new_test_scenes)}) scenes must cover all scenes ({len(scenes)})"
    with open(public / "train_data" / "scene.json", "w") as f:
        json.dump(new_train_scenes, f)
    with open(public / "test_data" / "scene.json", "w") as f:
        json.dump(new_test_scenes, f)

    """
    Copy sensor.json
    """
    # sensor.json is a list of sensors used in the dataset (10 sensors in the raw train set).
    # For simplicity, we'll just copy the full list of sensors in both the new train and new test sets.
    shutil.copyfile(
        src=raw / "train_data" / "sensor.json", dst=public / "train_data" / "sensor.json"
    )
    shutil.copyfile(
        src=raw / "train_data" / "sensor.json", dst=public / "test_data" / "sensor.json"
    )

    """
    Copy visibility.json
    """
    # visibility.json is a list of 4 visibility classes describing how visible an annotated object is in a given sample.
    # Both train and test use the same visibility classes, so we'll just copy these to the new train and new test sets.
    shutil.copyfile(
        src=raw / "train_data" / "visibility.json", dst=public / "train_data" / "visibility.json"
    )
    shutil.copyfile(
        src=raw / "train_data" / "visibility.json", dst=public / "test_data" / "visibility.json"
    )

    """
    Split sample_annotation.json
    """
    # sample_annotation.json is the full list of object annotations (bounding boxes) from all samples,
    # and will be split following the sample split.
    with open(raw / "train_data" / "sample_annotation.json") as f:
        sample_annotations = json.load(f)
    logger.info(f"Found {len(sample_annotations)} train sample annotations")
    new_train_sample_annotations, new_test_sample_annotations = [], []
    for sample_annotation in sample_annotations:
        sample_token = sample_annotation["sample_token"]
        if sample_token in new_train_df["Id"].values:
            new_train_sample_annotations.append(sample_annotation)
        elif sample_token in new_test_df["Id"].values:
            new_test_sample_annotations.append(sample_annotation)
        else:
            raise ValueError(
                f"Sample annotation token {sample_token} doesn't belong to either new train or new test set"
            )
    logger.info(
        f"New train sample annotations: {len(new_train_sample_annotations)}, new test sample annotations: {len(new_test_sample_annotations)}"
    )
    assert len(new_train_sample_annotations) + len(new_test_sample_annotations) == len(
        sample_annotations
    ), f"New train ({len(new_train_sample_annotations)}) and test ({len(new_test_sample_annotations)}) sample annotations must cover all sample annotations ({len(sample_annotations)})"
    with open(public / "train_data" / "sample_annotation.json", "w") as f:
        json.dump(new_train_sample_annotations, f)
    # NOTE: don't export (public / "test_data" / "sample_annotation.json") since the test set doesn't provide sample annotations

    """
    Split instance.json
    """
    # instance.json is a list of object instances (e.g. the same pedestrian appearing in contiguous frames),
    # and will be split following the sample_annotation.json split
    with open(raw / "train_data" / "instance.json") as f:
        instances = json.load(f)
    logger.info(f"Found {len(instances)} train instances")
    new_train_instance_ids = set([sa["instance_token"] for sa in new_train_sample_annotations])
    new_test_instance_ids = set([sa["instance_token"] for sa in new_test_sample_annotations])
    new_train_instances, new_test_instances = [], []
    for instance in instances:
        if instance["token"] in new_train_instance_ids:
            new_train_instances.append(instance)
        elif instance["token"] in new_test_instance_ids:
            new_test_instances.append(instance)
        else:
            raise ValueError(
                f"Instance {instance['token']} doesn't belong to either new train or new test set"
            )
    logger.info(
        f"New train instances: {len(new_train_instances)}, new test instances: {len(new_test_instances)}"
    )
    assert len(new_train_instances) + len(new_test_instances) == len(
        instances
    ), f"New train ({len(new_train_instances)}) and test ({len(new_test_instances)}) instances must cover all instances ({len(instances)})"
    with open(public / "train_data" / "instance.json", "w") as f:
        json.dump(new_train_instances, f)
    # NOTE: don't export (public / "test_data" / "instance.json") since the test set doesn't provide instance annotations

    """
    Copy over the heavy image and lidar data
    """
    (public / "test_images").mkdir(parents=True, exist_ok=True)
    (public / "train_images").mkdir(parents=True, exist_ok=True)
    (public / "test_lidar").mkdir(parents=True, exist_ok=True)
    (public / "train_lidar").mkdir(parents=True, exist_ok=True)
    if DEV_MODE:
        sample_data = sample_data[:100]  # Just copy a few samples for testing
    num_train_images, num_test_images = 0, 0
    num_train_lidar, num_test_lidar = 0, 0
    for sample_datum in tqdm(sample_data, desc="Copying images and lidar data"):
        filename = Path(
            sample_datum["filename"]
        ).name  # `filename` looks like "images/host-a011_cam2_1233689008717605006.jpeg", but we don't use that parent directory

        is_test = sample_datum["sample_token"] in new_test_df["Id"].values

        if sample_datum["fileformat"] == "jpeg":
            assert filename.endswith("jpeg"), f"Expected .jpeg, but got {filename}"
            src_file = raw / "train_images" / filename
            if not src_file.exists():
                raise FileNotFoundError(f"{src_file} does not exist")
            # Image
            if is_test:
                dst_file = public / "test_images" / filename
                if dst_file.exists():
                    logger.warning(f"Copying file to {dst_file}, but file already exists!")
                else:
                    shutil.copyfile(src=src_file, dst=dst_file)
                    num_test_images += 1
            else:
                dst_file = public / "train_images" / filename
                if dst_file.exists():
                    logger.warning(f"Copying file to {dst_file}, but file already exists!")
                else:
                    shutil.copyfile(src=src_file, dst=dst_file)
                    num_train_images += 1
        elif sample_datum["fileformat"] == "bin":
            assert filename.endswith("bin"), f"Expected .bin, but got {filename}"
            src_file = raw / "train_lidar" / filename
            if not src_file.exists():
                raise FileNotFoundError(f"{src_file} does not exist")
            # Lidar
            if is_test:
                dst_file = public / "test_lidar" / filename
                if dst_file.exists():
                    logger.warning(f"Copying file to {dst_file}, but file already exists!")
                else:
                    shutil.copyfile(src=src_file, dst=dst_file)
                    num_test_lidar += 1
            else:
                dst_file = public / "train_lidar" / filename
                if dst_file.exists():
                    logger.warning(f"Copying file to {dst_file}, but file already exists!")
                else:
                    shutil.copyfile(src=src_file, dst=dst_file)
                    num_train_lidar += 1
        else:
            raise ValueError(
                f"Unexpected `fileformat` in sample data: {sample_datum['fileformat']}"
            )
    assert num_train_images + num_test_images + num_train_lidar + num_test_lidar == len(
        set(sample_datum["filename"] for sample_datum in sample_data)
    ), f"Expected image and lidar samples for new train/test to cover all samples ({len(sample_data)})"
    assert num_train_images == len(
        list((public / "train_images").glob("*.jpeg"))
    ), f"Expected {num_train_images} train images, but got {len(list((public / 'train_images').glob('*.jpeg')))}"
    assert num_test_images == len(
        list((public / "test_images").glob("*.jpeg"))
    ), f"Expected {num_test_images} test images, but got {len(list((public / 'test_images').glob('*.jpeg')))}"
    assert num_train_lidar == len(
        list((public / "train_lidar").glob("*.bin"))
    ), f"Expected {num_train_lidar} train lidar files, but got {len(list((public / 'train_lidar').glob('*.bin')))}"
    assert num_test_lidar == len(
        list((public / "test_lidar").glob("*.bin"))
    ), f"Expected {num_test_lidar} test lidar files, but got {len(list((public / 'test_lidar').glob('*.bin')))}"
