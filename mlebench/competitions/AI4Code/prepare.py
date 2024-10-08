import json
import shutil
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from mlebench.utils import read_csv


# Create train, test from train split
def create_train_test_split(train_ancestors_df: pd.DataFrame, test_size: int):
    """
    On Kaggle, a user may "fork" (that is, copy) the notebook of another user to create their own version.
    `train_ancestors_df` contains information about the ancestry of each notebook in the training set.

    To create the test split, we select rows from the train set that don't share ancestors with any others
    so that there aren't any relatives in the train set. The train split is the remaining rows.
    """
    group_by_ancestor = {}
    for _, row in tqdm(
        train_ancestors_df.iterrows(), desc="Grouping by Ancestor", total=len(train_ancestors_df)
    ):
        ancestor_id = row["ancestor_id"]
        if ancestor_id not in group_by_ancestor:
            group_by_ancestor[ancestor_id] = []
        group_by_ancestor[ancestor_id].append(row)

    train_ids = []
    test_ids = []
    num_test_ids = 0
    for ancestor_id, rows in tqdm(group_by_ancestor.items(), desc="Splitting on Ancestors"):
        if num_test_ids + len(rows) <= test_size:
            test_ids.extend(([row["id"] for row in rows]))
            num_test_ids += len(rows)
        else:
            train_ids.extend([row["id"] for row in rows])

    assert len(test_ids) == test_size
    assert len(train_ids) == len(train_ancestors_df) - test_size

    return train_ids, test_ids


def prepare(raw: Path, public: Path, private: Path):
    train_ancestors_df = read_csv(raw / "train_ancestors.csv")
    # Shuffle the train_ancestors_df to ensure our split is random
    train_ancestors_df = train_ancestors_df.sample(frac=1, random_state=0).reset_index(drop=True)
    new_train_ids, new_test_ids = create_train_test_split(train_ancestors_df, test_size=20000)

    # Copy json files to public
    (public / "train").mkdir(parents=True, exist_ok=True)
    for train_id in tqdm(new_train_ids, desc="Copying train json files"):
        shutil.copy(raw / "train" / f"{train_id}.json", public / "train" / f"{train_id}.json")
    (public / "test").mkdir(parents=True, exist_ok=True)
    for test_id in tqdm(new_test_ids, desc="Copying test json files"):
        shutil.copy(raw / "train" / f"{test_id}.json", public / "test" / f"{test_id}.json")

    # Generate answers for train and test
    train_orders = read_csv(raw / "train_orders.csv")
    # Answers for new train
    train_orders_new = train_orders[train_orders["id"].isin(new_train_ids)]
    train_orders_new.to_csv(public / "train_orders.csv", index=False)
    # Answers for new test
    test_orders_new = train_orders[train_orders["id"].isin(new_test_ids)]
    test_orders_new.to_csv(private / "test_orders.csv", index=False)

    # Make new train_ancestors.csv, excluding the new_test_ids
    train_ancestors_df = train_ancestors_df[~train_ancestors_df["id"].isin(new_test_ids)]
    train_ancestors_df.to_csv(public / "train_ancestors.csv", index=False)

    # Create sample submission (use the given order without changing it)
    sample_submission_rows = []
    for sample_id in tqdm(test_orders_new["id"], desc="Creating sample submission"):
        # Get cell order from json file
        with open(public / "test" / f"{sample_id}.json") as f:
            json_data = json.load(f)
            cell_order = list(json_data["cell_type"].keys())
        sample_submission_rows.append({"id": sample_id, "cell_order": " ".join(cell_order)})
    sample_submission = pd.DataFrame(sample_submission_rows)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    assert len(sample_submission) == len(
        new_test_ids
    ), "Sample submission length does not match test length."
