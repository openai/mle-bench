import shutil
from itertools import islice
from pathlib import Path

import bson
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    dev_mode = False

    def read_ids_and_category_ids(file_path: Path) -> pd.DataFrame:
        data = bson.decode_file_iter(open(file_path, "rb"))

        records = []

        for c, d in enumerate(tqdm(data, desc="Reading BSON data")):
            records.append({"_id": d["_id"], "category_id": d["category_id"]})

        return pd.DataFrame(records)

    def filter_bson_by_ids(
        bson_file_path: Path,
        ids: set,
        write_path: Path,
        exclude_cols: list = [],
        chunk_size=1000,
        max_rows=None,
    ):
        """
        Filters a BSON file by a set of IDs and writes the filtered data to a new BSON file.
        The original _id is replaced with a new _id starting from 0 and incrementing by 1.

        Args:
            bson_file_path (Path): Path to the input BSON file.
            ids (set): Set of IDs to filter by.
            write_path (Path): Path to the output BSON file.
            exclude_cols (list): List of columns to exclude from the output.
            max_rows (int, optional): Maximum number of rows to write to the output file.
        """
        data = bson.decode_file_iter(open(bson_file_path, "rb"))
        num_written_rows = 0

        with open(write_path, "wb") as f:
            for record in tqdm(data, desc="Filtering BSON data"):
                if record["_id"] in ids:
                    for col in exclude_cols:
                        if col in record:
                            del record[col]
                    num_written_rows += 1
                    f.write(bson.BSON.encode(record))

                if num_written_rows % chunk_size == 0:
                    f.flush()

                if max_rows is not None and num_written_rows >= max_rows:
                    break

    # Create train, test from train split. Original train.bson contains 7,069,896 rows. Original test.bson contains 1,768,182 rows.
    old_train = read_ids_and_category_ids(raw / "train.bson")

    # Ensure rows in train_example remain in new_train
    new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)
    answers = answers.sort_values(by="_id")

    # Create sample submission
    sample_submission = answers[["_id"]]
    sample_submission["category_id"] = 1000010653

    # Checks
    assert len(new_train) + len(answers) == len(
        old_train
    ), f"The length of new_train and answers combined should be equal to the original length of old_train. Got {len(new_train) + len(answers)} and {len(old_train)}"
    assert set(new_train["_id"]).isdisjoint(
        set(answers["_id"])
    ), "new_train and answers should not have any _ids in common"
    assert sample_submission.columns.tolist() == [
        "_id",
        "category_id",
    ], f"sample_submission should have columns _id and category_id. Got {sample_submission.columns.tolist()}"

    # Write new files
    answers.to_csv(private / "answers.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    filter_bson_by_ids(
        bson_file_path=(
            raw / "train_example.bson" if dev_mode else raw / "train.bson"
        ),  # train_example.bson is the first 100 rows of train.bson
        ids=set(new_train["_id"]),
        write_path=public / "train.bson",
    )
    filter_bson_by_ids(
        bson_file_path=raw / "train_example.bson" if dev_mode else raw / "train.bson",
        ids=set(answers["_id"]),
        write_path=public / "test.bson",
        exclude_cols=["category_id"],  # removes category_id for test.bson
    )

    # Write new train_example.bson which is the first 100 rows of the new train.bson
    filter_bson_by_ids(
        bson_file_path=(
            raw / "train_example.bson" if dev_mode else raw / "train.bson"
        ),  # train_example.bson is the first 100 rows of train.bson
        ids=set(new_train["_id"]),
        write_path=public / "train_example.bson",
        max_rows=100,
    )

    def is_valid_bson_file(file_path: Path, chunk_size: int = 10000):
        try:
            with open(file_path, "rb") as f:
                data_iter = bson.decode_file_iter(f)
                for chunk in tqdm(
                    iter(lambda: list(islice(data_iter, chunk_size)), []),
                    desc=f"Validating {file_path.name}",
                ):
                    pd.DataFrame(chunk)  # Attempt to create a DataFrame from the chunk
        except Exception as e:
            return False

        return True

    # Check train.bson
    assert is_valid_bson_file(public / "train.bson"), f"Couldn't parse `train.bson` as a bson file!"

    # Check test.bson
    assert is_valid_bson_file(public / "test.bson"), f"Couldn't parse `test.bson` as a bson file!"

    # Copy over other files
    shutil.copy(raw / "category_names.csv", public / "category_names.csv")

    actual_new_train = read_ids_and_category_ids(public / "train.bson")
    actual_new_train_example = read_ids_and_category_ids(public / "train_example.bson")

    assert actual_new_train.iloc[:100].equals(
        actual_new_train_example
    ), f"The first 100 rows of `train.bson` should be the same as `train_example.bson`"
