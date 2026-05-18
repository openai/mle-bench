import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import extract, get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Extract images so we can split the train images
    extract(raw / "train_images.zip", raw / "train")

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")

    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Save the new train and test
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # Copy images
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in tqdm(new_train["image_id"], desc="Copying train images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    for file_id in tqdm(new_test["image_id"], desc="Copying test images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    assert len(list(public.glob("train/*.jpg"))) == len(new_train)
    assert len(list(public.glob("test/*.jpg"))) == len(new_test)
    assert len(new_train) + len(new_test) == len(old_train)

    # Create zips of the images
    logger.info("Re-zipping up new image directories...")
    shutil.make_archive(str(public / "train_images"), "zip", public / "train")
    shutil.make_archive(str(public / "test_images"), "zip", public / "test")
    # Remove the directories for consistency with the kaggle data
    shutil.rmtree(public / "train")
    shutil.rmtree(public / "test")

    # Copy unicode_translation
    shutil.copyfile(
        src=raw / "unicode_translation.csv",
        dst=public / "unicode_translation.csv",
    )

    assert (public / "train_images.zip").is_file()
    assert (public / "test_images.zip").is_file()
    assert (public / "unicode_translation.csv").is_file()

    # Make sample submission for new test set
    sample_submission = new_test.copy()
    # Same guess for all, as in original sample submission
    sample_submission["labels"] = "U+003F 1 1 U+FF2F 2 2"
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Make a gold submission in private/ (useful for testing)
    # This submission takes the "class x y w h" labels from test and converts them to
    # "class x+1 y+1" labels (the +1 moves the coord into the  bbox, so that the metric picks it up)
    submission_labels = []
    for label in new_test["labels"]:
        # Labels have the form "class x y w h class x y w h class x y w h ... "
        label = label.split()
        new_label = []
        assert len(label) % 5 == 0
        classes, xs, ys = label[0::5], label[1::5], label[2::5]
        # +1 to xs and ys
        xs = [str(int(x) + 1) for x in xs]
        ys = [str(int(y) + 1) for y in ys]
        new_label = [f"{c} {x} {y}" for c, x, y in zip(classes, xs, ys)]
        submission_labels.append(" ".join(new_label))
    gold_submission = new_test.copy()
    gold_submission["labels"] = submission_labels
    gold_submission.to_csv(private / "gold_submission.csv", index=False)

    assert len(gold_submission) == len(new_test)
    assert len(gold_submission) == len(sample_submission)
