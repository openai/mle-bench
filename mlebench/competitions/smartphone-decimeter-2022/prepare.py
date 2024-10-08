import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_date(s: str) -> str:
    """Gets date from string in the format YYYY-MM-DD-X where `X` is an arbitrary string."""

    split = s.split("-")

    assert (
        len(split) >= 3
    ), f"Expected the string to have at least 3 parts separated by `-`. Got {len(split)} parts."

    year, month, day = split[:3]

    assert (
        isinstance(year, str) and year.isdigit()
    ), f"Expected the year to be a string of digits. Got {year} instead."

    assert (
        isinstance(month, str) and month.isdigit()
    ), f"Expected the month to be a string of digits. Got {month} instead."

    assert (
        isinstance(day, str) and day.isdigit()
    ), f"Expected the day to be a string of digits. Got {day} instead."

    date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return date


def prepare(raw: Path, public: Path, private: Path) -> None:
    old_train_ids = sorted([folder.name for folder in (raw / "train").glob("*") if folder.is_dir()])
    dates = sorted(set([get_date(s) for s in old_train_ids]))
    new_train_dates, new_test_dates = train_test_split(dates, test_size=0.1, random_state=0)

    assert (
        len(new_train_dates) >= 1
    ), "Expected the new train set to have at least one date. Got 0 dates."

    assert (
        len(new_test_dates) >= 1
    ), "Expected the new test set to have at least one date. Got 0 dates."

    new_train_ids = sorted([i for i in old_train_ids if get_date(i) in new_train_dates])
    new_test_ids = sorted([i for i in old_train_ids if get_date(i) in new_test_dates])

    assert len(set(new_train_ids).intersection(set(new_test_ids))) == 0, (
        f"Expected the new train and test instances to be disjoint. Got an intersection of "
        f"{set(new_train_ids).intersection(set(new_test_ids))}."
    )

    assert len(new_train_ids) + len(new_test_ids) == len(old_train_ids), (
        f"Expected the number of new train and test instances to sum up to the number of old train "
        f"instances. Got {len(new_train_ids)} new train instances and {len(new_test_ids)} new test "
        f"instances which sum to {len(new_train_ids) + len(new_test_ids)} instead of "
        f"{len(old_train_ids)}."
    )

    assert set(new_train_ids).intersection(new_test_ids) == set(), (
        f"Expected the new train and test instances to be disjoint. Got an intersection of "
        f"{set(new_train_ids).intersection(new_test_ids)}."
    )

    for new_train_id in new_train_ids:
        shutil.copytree(
            src=raw / "train" / new_train_id,
            dst=public / "train" / new_train_id,
        )

    for new_test_id in new_test_ids:
        shutil.copytree(
            src=raw / "train" / new_test_id,
            dst=public / "test" / new_test_id,
        )

    # Construct test set by concatenating all ground truth csvs for the test journeys
    dfs = []

    for fpath in sorted((public / "test").rglob("ground_truth.csv")):
        drive_id = fpath.parent.parent.name
        phone_id = fpath.parent.name

        assert (
            drive_id in new_test_ids
        ), f"Expected the drive {drive_id} to be one of the new test instances. Got {drive_id} instead."

        raw_df = pd.read_csv(fpath)
        df = raw_df.copy()
        df.loc[:, "tripId"] = f"{drive_id}-{phone_id}"
        df = df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
        dfs.append(df)

    new_test = pd.concat(dfs, ignore_index=True)
    new_test.to_csv(private / "test.csv", index=False)

    for fpath in (public / "test").rglob("ground_truth.csv"):
        fpath.unlink()  # don't include ground truth in public test data

    shutil.copytree(
        src=raw / "metadata",
        dst=public / "metadata",
    )

    actual_journey_ids = set(["-".join(s.split("-")[:-1]) for s in new_test["tripId"]])

    assert len(actual_journey_ids) == len(new_test_ids), (
        f"Expected the new test instances to have {len(new_test_ids)} unique trip IDs. Got "
        f"{len(new_test['tripId'].unique())} unique trip IDs."
    )

    sample_submission = new_test.copy()
    sample_submission.loc[:, "LatitudeDegrees"] = 37.904611315634504
    sample_submission.loc[:, "LongitudeDegrees"] = -86.48107806249548

    assert len(sample_submission) == len(new_test), (
        f"Expected the sample submission to have the same number of instances as the new test "
        f"instances. Got {len(sample_submission)} instances in the sample submission and "
        f"{len(new_test)} new test instances."
    )

    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    assert sorted(list(public.glob("train/*"))) == sorted(
        set([public / "train" / drive_id for drive_id in new_train_ids])
    ), "Expected the public train directory to contain the new train instances."

    assert sorted(list(public.glob("test/*"))) == sorted(
        set([public / "test" / drive_id for drive_id in new_test_ids])
    ), "Expected the public test directory to contain the new test instances."

    assert (
        len(list((public / "test").rglob("ground_truth.csv"))) == 0
    ), "Expected the public test directory to not contain any ground truth files."

    assert len(list((public / "train").rglob("ground_truth.csv"))) >= len(new_train_ids), (
        "Expected the public train directory to contain at least one ground truth file per new "
        "train instance."
    )
