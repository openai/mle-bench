import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    rng = random.Random(0)

    # there are two subsets of training data:

    # 1. one of instances that have bounding boxes
    # 2. one of instances that have segmentations

    # we need to preserve the ratios of the sizes of these subsets to the total train samples

    # additionally, there is an overlap between the two subsets
    # we need to preserve this overlap

    DEV = False
    old_train = read_csv(raw / "train.csv")

    num_old_train = len(old_train)
    if DEV:
        DEV_RATIO = 0.175
        num_old_train = round(DEV_RATIO * num_old_train)

    # 2019 train folders (StudyInstanceUIDs), 1500 test folders, 2019 / (1500 + 2019) ~ 0.60 original train ratio
    # each folder has ~ 300 images
    # We use 0.1 ratio to avoid taking too many samples out of train
    TRAIN_RATIO = 0.1
    num_train_samples = round(num_old_train * TRAIN_RATIO)

    # bboxes
    old_train_bboxes = read_csv(raw / "train_bounding_boxes.csv")
    if DEV:
        old_train_bboxes = old_train_bboxes.sample(frac=DEV_RATIO, random_state=0)

    old_train_bbox_ids = sorted(old_train_bboxes["StudyInstanceUID"].unique())
    old_num_train_bbox_ids = len(old_train_bbox_ids)  # 235
    new_num_train_bbox_ids = round(old_num_train_bbox_ids * TRAIN_RATIO)

    # segmentations
    old_train_segmentation_path = raw / "segmentations"
    old_train_segmentation_ids = sorted([f.stem for f in old_train_segmentation_path.glob("*.nii")])
    if DEV:
        old_train_segmentation_ids = rng.sample(
            old_train_segmentation_ids, round(DEV_RATIO * len(old_train_segmentation_ids))
        )
    old_num_train_segmentation_ids = len(old_train_segmentation_ids)  # 87
    new_num_train_segmentation_ids = round(old_num_train_segmentation_ids * TRAIN_RATIO)

    # overlap: list of StudyInstanceUIDs that have both bounding boxes and segmentations
    old_overlap_ids = [uid for uid in old_train_bbox_ids if uid in old_train_segmentation_ids]
    old_num_overlap = len(old_overlap_ids)  # 40
    new_num_overlap = round(old_num_overlap * TRAIN_RATIO)

    # start populating new train by picking the overlap instances
    # sample new_num_overlap instances from the overlap randomly
    new_overlap_ids = rng.sample(old_overlap_ids, new_num_overlap)
    new_bboxes_ids = new_overlap_ids.copy()
    new_segmentations_ids = new_overlap_ids.copy()
    new_train_ids = new_overlap_ids.copy()

    # add the `new_num_train_segmentation_ids - new_num_overlap`, that are not in the overlap
    additional_segmentation_ids = rng.sample(
        [uid for uid in old_train_segmentation_ids if uid not in old_overlap_ids],
        new_num_train_segmentation_ids - new_num_overlap,
    )
    new_segmentations_ids += additional_segmentation_ids
    new_train_ids += additional_segmentation_ids

    # add the (`new_num_train_bbox_ids - num_num_overlap`) segmentations, that are not in the overlap
    additional_bbox_ids = rng.sample(
        [uid for uid in old_train_bbox_ids if uid not in old_overlap_ids],
        new_num_train_bbox_ids - new_num_overlap,
    )
    new_bboxes_ids += additional_bbox_ids
    new_train_ids += additional_bbox_ids

    if DEV:
        # old train has whatever is currently in new_train_ids
        # + a random sample of the rest, s.t. its 15% of the original train
        dev_old_train_ids = new_train_ids + rng.sample(
            [uid for uid in old_train["StudyInstanceUID"] if uid not in new_train_ids],
            num_old_train - len(new_train_ids),
        )
        old_train = old_train[old_train["StudyInstanceUID"].isin(dev_old_train_ids)].copy()

    # then, fill the rest of the new train.
    new_train_ids += rng.sample(
        [uid for uid in old_train["StudyInstanceUID"] if uid not in new_train_ids],
        num_train_samples - len(new_train_ids),
    )

    train = old_train[old_train["StudyInstanceUID"].isin(new_train_ids)].copy()
    train.to_csv(public / "train.csv", index=False)

    train_bboxes = old_train_bboxes[
        old_train_bboxes["StudyInstanceUID"].isin(new_bboxes_ids)
    ].copy()
    train_bboxes.to_csv(public / "train_bounding_boxes.csv", index=False)

    answers = old_train[~old_train["StudyInstanceUID"].isin(new_train_ids)].copy()
    # columns become rows for the test and sample submission, so also for answers
    answers = answers.melt(
        id_vars="StudyInstanceUID", var_name="prediction_type", value_name="fractured"
    )
    answers["row_id"] = answers["StudyInstanceUID"] + "_" + answers["prediction_type"]
    answers.to_csv(private / "answers.csv", index=False)

    sample_submission = answers[["row_id", "fractured"]].copy()
    sample_submission["fractured"] = 0.5
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    public_test = answers.drop(columns=["fractured"]).copy()
    public_test.to_csv(public / "test.csv", index=False)

    # assert that the melting worked
    assert answers["StudyInstanceUID"].nunique() * 8 == len(
        answers
    ), "Melting failed, incorrect length"
    assert answers.columns.tolist() == [
        "StudyInstanceUID",
        "prediction_type",
        "fractured",
        "row_id",
    ], "Melting went wrong, columns are wrong"

    # column checks
    train_cols = ["StudyInstanceUID", "patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    assert train.columns.tolist() == train_cols, "Train columns are wrong"
    bbox_cols = ["StudyInstanceUID", "x", "y", "width", "height", "slice_number"]
    assert train_bboxes.columns.tolist() == bbox_cols, "Bounding box columns are wrong"
    test_cols = ["StudyInstanceUID", "prediction_type", "row_id"]
    assert public_test.columns.tolist() == test_cols, "Test columns are wrong"
    submission_cols = ["row_id", "fractured"]
    assert sample_submission.columns.tolist() == submission_cols, "Submission columns are wrong"

    # Check that the correct number of training samples is selected
    assert len(new_train_ids) == round(len(old_train) * TRAIN_RATIO), (
        "Incorrect number of training samples."
        " The number of `new_train_ids` doesn't match the expected number given the `TRAIN_RATIO`."
    )
    assert len(train) + answers["StudyInstanceUID"].nunique() == len(old_train), (
        "Incorrect number of training samples."
        " New train and test splits don't sum to the length of the original train set."
    )

    # Check that the correct number of bounding box samples is selected
    assert len(new_bboxes_ids) == round(
        len(old_train_bbox_ids) * TRAIN_RATIO
    ), "Incorrect number of bounding box samples"

    # Check that the correct number of segmentation samples is selected
    assert len(new_segmentations_ids) == round(
        len(old_train_segmentation_ids) * TRAIN_RATIO
    ), "Incorrect number of segmentation samples"

    # Check that the overlap is preserved
    assert len(new_overlap_ids) == round(
        len(old_overlap_ids) * TRAIN_RATIO
    ), "Incorrect overlap preservation"

    # check that test and train dont share study instance ids
    assert set(train["StudyInstanceUID"]).isdisjoint(
        set(public_test["StudyInstanceUID"].unique())
    ), "Train and test share study instance ids"

    # Now that splitting is done, copy over images accordingly
    (public / "segmentations").mkdir(exist_ok=True)

    for file_id in tqdm(
        new_segmentations_ids, desc="Copying segmentations", total=len(new_segmentations_ids)
    ):
        shutil.copyfile(
            src=old_train_segmentation_path / f"{file_id}.nii",
            dst=public / "segmentations" / f"{file_id}.nii",
        )

    (public / "train_images").mkdir(exist_ok=True)
    for study_id in tqdm(
        train["StudyInstanceUID"],
        desc="Copying train images",
        total=len(train),
        unit="StudyInstance",
    ):
        shutil.copytree(
            src=raw / "train_images" / study_id,
            dst=public / "train_images" / study_id,
            dirs_exist_ok=True,
        )
    (public / "test_images").mkdir(exist_ok=True)
    for study_id in tqdm(
        public_test["StudyInstanceUID"].unique(),
        desc="Copying test images",
        total=public_test["StudyInstanceUID"].nunique(),
        unit="StudyInstance",
    ):
        shutil.copytree(
            src=raw / "train_images" / study_id,
            dst=public / "test_images" / study_id,
            dirs_exist_ok=True,
        )

    assert len(list((public / "segmentations").glob("*.nii"))) == len(
        new_segmentations_ids
    ), "Incorrect number of segmentations copied"
    # check that all the right image directories are copied
    assert len(list((public / "train_images").glob("*"))) == len(
        train
    ), "Incorrect number of train images copied"

    assert (
        len(list((public / "test_images").glob("*"))) == public_test["StudyInstanceUID"].nunique()
    ), "Incorrect number of test images copied"
