from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import get_logger

logger = get_logger(__name__)


def extract_string(document_text: str, start_token_idx: int, end_token_idx: int) -> str:
    document_tokens = document_text.split(" ")
    extract_tokens = document_tokens[start_token_idx:end_token_idx]
    return " ".join(extract_tokens)


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    train_file = "simplified-nq-train.jsonl"

    logger.info("Counting lines in train file...")
    with open(raw / train_file, "r") as f:
        n_lines = sum(1 for _ in f)
    logger.info(f"Found {n_lines} lines in train file.")

    # Read data in chunks to avoid memory issues
    train_ids, test_ids = [], []
    lightweight_test = []  # We'll use this to create a gold submission later
    with tqdm(total=n_lines, desc="Splitting data") as pbar:
        for df in pd.read_json(raw / train_file, orient="records", lines=True, chunksize=1_000):
            # Convert IDs to strings, Kaggle.com is inconsistent about this but strings make more sense
            df["example_id"] = df["example_id"].astype(str)
            new_train, new_test = train_test_split(df, test_size=0.1, random_state=0)

            keys_to_keep = [
                "example_id",
                "question_text",
                "document_text",
                "long_answer_candidates",
            ]
            new_test_without_labels = new_test.copy()[keys_to_keep]

            # Append lines to new train and test
            with open(public / "simplified-nq-train.jsonl", "a") as f:
                f.write(new_train.to_json(orient="records", lines=True))
            with open(private / "test.jsonl", "a") as f:
                f.write(new_test.to_json(orient="records", lines=True))
            with open(public / "simplified-nq-test.jsonl", "a") as f:
                f.write(new_test_without_labels.to_json(orient="records", lines=True))

            train_ids.extend(new_train["example_id"].tolist())
            test_ids.extend(new_test["example_id"].tolist())
            lightweight_test.append(
                new_test.copy()[["example_id", "question_text", "annotations"]]
            )  # For gold submission
            pbar.update(len(df))

    lightweight_test = pd.concat(lightweight_test, ignore_index=True)

    assert len(train_ids) + len(test_ids) == n_lines
    assert len(lightweight_test) == len(test_ids)

    # Create a gold submission with columns "example_id", "PredictionString"
    gold_rows = []
    for idx, sample in tqdm(
        lightweight_test.iterrows(), total=len(lightweight_test), desc="Creating gold submission"
    ):
        sample = sample.to_dict()
        assert len(sample["annotations"]) == 1
        annotation = sample["annotations"][0]

        # Create short answer

        # Multiple answers are possible: yes_no_answer or one of short_answers
        # We just take the first one
        if annotation["yes_no_answer"] != "NONE":
            answer = annotation["yes_no_answer"]
        elif len(annotation["short_answers"]) > 0:
            start_token = annotation["short_answers"][0]["start_token"]
            end_token = annotation["short_answers"][0]["end_token"]
            answer = f"{start_token}:{end_token}"
        else:
            answer = ""

        logger.debug(f"q: {sample['question_text']}")
        logger.debug(f"a: {answer}")
        logger.debug("")

        gold_rows.append(
            {"example_id": f"{sample['example_id']}_short", "PredictionString": answer}
        )

        # Create long answer

        if annotation["long_answer"]["start_token"] != -1:
            start_token = annotation["long_answer"]["start_token"]
            end_token = annotation["long_answer"]["end_token"]
            answer = f"{start_token}:{end_token}"
        else:
            answer = ""

        logger.debug(f"q: {sample['question_text']}")
        logger.debug(f"a: {answer}")
        logger.debug("")

        gold_rows.append({"example_id": f"{sample['example_id']}_long", "PredictionString": answer})

    gold_submission = pd.DataFrame(gold_rows)
    gold_submission.to_csv(private / "gold_submission.csv", index=False)

    # Sample submission
    sample_submission = gold_submission.copy()
    sample_submission["PredictionString"] = ""
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    assert len(gold_submission) == 2 * len(test_ids)
    assert len(sample_submission) == 2 * len(test_ids)
