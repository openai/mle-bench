import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from mlebench.utils import extract, get_logger

np_rng = np.random.RandomState(0)

logger = get_logger(__name__)


def count_lines_in_file(file_path):
    line_count = 0
    with open(file_path, "r") as file:
        for _line in file:
            line_count += 1
    return line_count


def compress_file_to_zip(src_file: Path, zip_file: Path):
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(src_file, arcname=src_file.name)


def remove_random_word(sentence: str) -> str:
    """
    Remove a random 'word' (sequence of characters, delimited by whitespace) from a sentence.
    Does not remove first or last words.

    Punctuation counts as a word, and is already separated by whitespace.
    """
    words = sentence.split()
    index = np_rng.randint(1, len(words) - 1)
    return " ".join(words[:index] + words[index + 1 :])


def prepare(raw: Path, public: Path, private: Path):
    logger.info("Extracting raw / train_v2.txt.zip")
    extract(raw / "train_v2.txt.zip", raw)

    # computed this ahead of time
    total_lines = 30301028

    with (
        open(raw / "train_v2.txt", "r") as old_train,
        open(public / "train_v2.txt", "w") as public_train,
        open(public / "test_v2.txt", "w") as public_test,
        open(private / "test.csv", "w") as private_test,
    ):
        public_test.write('"id","sentence"\n')
        private_test.write('"id","sentence"\n')
        line_count = 0
        test_count = 0
        train_count = 0
        # there is one sentence per line
        for sentence in tqdm(old_train, desc="Processing data", total=total_lines):
            # we will put ~0.01 of the data in test, the rest in train, matching kaggle's original split
            # some sentences only have 2 words, so can't remove a word -- keep them in train
            if np_rng.uniform() <= 0.01 and len(sentence.strip().split()) > 2:
                # get rid of linebreak and escape quotes
                sentence = sentence.strip().replace('"', '""')
                removed_word_sentence = remove_random_word(sentence)
                private_test.write(f'{test_count},"{sentence}"\n')
                public_test.write(f'{test_count},"{removed_word_sentence}"\n')
                test_count += 1
            else:
                public_train.write(sentence)
                train_count += 1
            line_count += 1
            if line_count >= total_lines:
                break

    # we will be compressing the public files (to match what's on kaggle.com)
    # so copy our sample submission to private so we have access to it
    shutil.copy(public / "test_v2.txt", private / "sample_submission.csv")

    # compress the public files
    logger.info("Compressing train_v2.txt")
    compress_file_to_zip(public / "train_v2.txt", public / "train_v2.txt.zip")
    logger.info("Compressing test_v2.txt")
    compress_file_to_zip(public / "test_v2.txt", public / "test_v2.txt.zip")
    # remove the original files
    (public / "train_v2.txt").unlink()
    (public / "test_v2.txt").unlink()

    # Checks
    assert not (public / "train_v2.txt").exists(), "public / 'train_v2.txt' should not exist"
    assert (public / "train_v2.txt.zip").exists(), "public / 'train_v2.txt.zip' should exist"
    assert not (public / "test_v2.txt").exists(), "public / 'test_v2.txt' should not exist"
    assert (public / "test_v2.txt.zip").exists(), "public / 'test_v2.txt.zip' should exist"

    private_test_line_count = count_lines_in_file(private / "test.csv")
    assert (
        # minus 2 to exclude header
        private_test_line_count - 1
        == test_count
    ), "private / 'test.csv' has incorrect number of lines"
    assert (
        count_lines_in_file(private / "sample_submission.csv") == private_test_line_count
    ), "private / 'sample_submission.csv' has incorrect number of lines"
    assert (
        test_count + train_count == total_lines
    ), "Expected the number of test samples and train samples to sum to the total number of samples in the original train file"
