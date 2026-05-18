import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import extract, read_csv


def prepare(raw: Path, public: Path, private: Path) -> None:
    extract(raw / "competition_data.zip", raw)

    old_train = read_csv(raw / "competition_data" / "train.csv")
    old_train = old_train.fillna("")

    # Original ratio is Train set - 4,000 samples; Test set - ~18,000 samples (82% ratio)
    # We use a 0.25 ratio to get number of test samples into thousand OOM
    new_train, new_test = train_test_split(old_train, test_size=0.25, random_state=0)

    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Some samples were lost when creating the new train and test sets!"

    new_train_ids = set(new_train["id"])
    new_test_ids = set(new_test["id"])

    assert new_train_ids.isdisjoint(
        new_test_ids
    ), "`id` is not disjoint between train and test sets"

    new_train.sort_values(by="id").to_csv(public / "train.csv", index=False)
    new_test.sort_values(by="id").to_csv(private / "test.csv", index=False)

    old_train_imgs = set((raw / "competition_data" / "train" / "images").glob("*.png"))

    assert len(old_train_imgs) == len(
        old_train
    ), "The number of images in the old train set doesn't match the number of training images!"

    new_train_imgs = set(img for img in old_train_imgs if img.stem in new_train_ids)
    new_test_imgs = set(img for img in old_train_imgs if img.stem in new_test_ids)

    assert new_train_imgs.isdisjoint(
        new_test_imgs
    ), "Images are not disjoint between train and test sets"

    assert len(new_train_imgs) + len(new_test_imgs) == len(
        old_train_imgs
    ), "Some images were lost when creating the new train and test sets!"

    (public / "train" / "images").mkdir(parents=True, exist_ok=True)
    (public / "train" / "masks").mkdir(parents=True, exist_ok=True)

    for fpath in new_train_imgs:
        shutil.copyfile(
            src=fpath,
            dst=public / "train" / "images" / fpath.name,
        )

        shutil.copyfile(
            src=raw / "competition_data" / "train" / "masks" / fpath.name,
            dst=public / "train" / "masks" / fpath.name,
        )

    (public / "test" / "images").mkdir(parents=True, exist_ok=True)

    for fpath in new_test_imgs:
        shutil.copyfile(
            src=fpath,
            dst=public / "test" / "images" / fpath.name,
        )

    sample_submission = new_test.drop(columns=["rle_mask"]).copy()
    sample_submission["rle_mask"] = "1 1"
    sample_submission.sort_values(by="id").to_csv(public / "sample_submission.csv", index=False)

    old_depths = read_csv(raw / "depths.csv")
    new_depths_mask = old_depths["id"].isin(new_train_ids)
    new_depths = old_depths[new_depths_mask]
    new_depths.sort_values(by="id").to_csv(public / "depths.csv", index=False)

    # Sanity checks

    assert (public / "train.csv").exists(), "`train.csv` doesn't exist!"
    assert (public / "sample_submission.csv").exists(), "`sample_submission.csv` doesn't exist!"
    assert (public / "depths.csv").exists(), "`depths.csv` doesn't exist!"
    assert (public / "train").exists(), "`train` directory doesn't exist!"
    assert (public / "test").exists(), "`test` directory doesn't exist!"
    assert (private / "test.csv").exists(), "`test.csv` doesn't exist!"

    actual_new_train_imgs = set(img.stem for img in (public / "train" / "images").glob("*.png"))
    actual_new_train_masks = set(img.stem for img in (public / "train" / "masks").glob("*.png"))

    assert len(actual_new_train_imgs) == len(
        new_train
    ), "The number of images in the train set doesn't match the number of training images!"

    assert len(actual_new_train_masks) == len(
        new_train
    ), "The number of masks in the train set doesn't match the number of training masks!"

    for new_train_id in new_train["id"]:
        assert (
            public / "train" / "images" / f"{new_train_id}.png"
        ).exists(), f"Expected `{new_train_id}.png` to exist in train images, but it doesn't!"

        assert (
            public / "train" / "masks" / f"{new_train_id}.png"
        ).exists(), f"Expected `{new_train_id}.png` to exist in train masks, but it doesn't!"

    actual_new_test_imgs = set(img.stem for img in (public / "test" / "images").glob("*.png"))

    assert not (
        public / "test" / "masks"
    ).exists(), f"Expected `public / test / masks` to not exist, but it does!"

    assert len(actual_new_test_imgs) == len(
        new_test
    ), "The number of images in the test set doesn't match the number of test images!"

    for new_test_id in new_test["id"]:
        assert (
            public / "test" / "images" / f"{new_test_id}.png"
        ).exists(), f"Expected `{new_test_id}.png` to exist in test images, but it doesn't!"

        assert not (
            public / "test" / "masks" / f"{new_test_id}.png"
        ).exists(), f"Expected `{new_test_id}.png` to exist in test masks, but it doesn't!"

    assert actual_new_train_imgs.isdisjoint(
        actual_new_test_imgs
    ), "Expected no overlap in images between the new train and test sets, but there is!"

    actual_sample_submission = read_csv(public / "sample_submission.csv")
    actual_new_test = read_csv(private / "test.csv")

    assert len(actual_sample_submission) == len(
        actual_new_test
    ), "The number of samples in the sample submission doesn't match the number of samples in the test set!"

    assert set(actual_sample_submission["id"]) == set(
        actual_new_test["id"]
    ), "The ids in the sample submission don't match the ids in the test set!"

    assert len(actual_new_test_imgs) == len(
        actual_new_test
    ), "The number of images in the test set doesn't match the number of test images!"

    assert (
        set(actual_new_test["id"]) == actual_new_test_imgs
    ), "The ids in the test set don't match the test images!"

    assert new_train.applymap(
        lambda x: isinstance(x, str)
    ).values.all(), "Not all elements in the DataFrame are strings!"

    assert new_test.applymap(
        lambda x: isinstance(x, str)
    ).values.all(), "Not all elements in the DataFrame are strings!"
