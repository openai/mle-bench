import os
import shutil
from pathlib import Path

import pandas as pd

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "transactions_train.csv")
    old_train["purchase_id"] = (
        old_train["customer_id"].astype(str)
        + "_"
        + old_train["article_id"].astype(str)
        + "_"
        + old_train["t_dat"].astype(str)
    )

    # The task is to predict what purchases will be made in the next 7 days.
    # To create our test set, we will take the purchases made in the last 7 days of the training set.
    old_train["t_dat_parsed"] = pd.to_datetime(
        old_train["t_dat"]
    )  # Parse t_dat to datetime in a new column
    max_date = old_train["t_dat_parsed"].max()  # Find the maximum date in the t_dat_parsed column
    old_train["in_last_7_days"] = old_train["t_dat_parsed"] >= (max_date - pd.Timedelta(days=7))
    new_train = old_train[
        old_train["in_last_7_days"] == False
    ].copy()  # Filter rows where t_dat_parsed is more than 7 days from the maximum date
    new_test = old_train[
        old_train["in_last_7_days"] == True
    ].copy()  # Filter rows where t_dat_parsed is within the last 7 days of the time series

    # Train/test checks
    assert (
        not new_test["purchase_id"].isin(new_train["purchase_id"]).any()
    ), "No purchase_ids should be shared between new_test and new_train"
    new_train = new_train.drop(columns=["purchase_id", "t_dat_parsed", "in_last_7_days"])
    new_test = new_test.drop(columns=["purchase_id", "t_dat_parsed"])

    # sample submission and answers differ because the task is predicting what articles each
    # customer will purchase in the 7-day period immediately after the training data ends. Customer
    # who did not make any purchase during that time are excluded from the scoring.

    # As such we can't put the exact customer ids from test set into the sample submission, as this
    # would leak which customers made purchases in the test set. Instead, we put _all_ the customer
    # ids in the sample submission, ask the user to predict for all of them, and then we will filter
    # out in grade.py the customers who did not make any purchases in the test set.

    # Answers, contains only customers that actually made purchases in the test period.
    answers = (
        new_test.groupby("customer_id")["article_id"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    # rename 'article_id' to 'prediction'
    answers = answers.rename(columns={"article_id": "prediction"})

    # Sample submission, which contains all customer ids.
    shutil.copyfile(
        src=raw / "sample_submission.csv",
        dst=public / "sample_submission.csv",
    )

    # Write CSVs
    # new_test.to_csv(private / "test.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)
    new_train.to_csv(public / "transactions_train.csv", index=False)

    # Copy files and images directory
    shutil.copyfile(
        src=raw / "articles.csv",
        dst=public / "articles.csv",
    )
    shutil.copyfile(
        src=raw / "customers.csv",
        dst=public / "customers.csv",
    )
    shutil.copytree(
        src=raw / "images",
        dst=public / "images",
        dirs_exist_ok=True,
    )

    # checks
    expected_train_columns = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    assert (
        new_train.columns.tolist() == expected_train_columns
    ), f"Unexcpected columns in new_train, expected {expected_train_columns}, got {new_train.columns.tolist()}"

    expected_answer_columns = ["customer_id", "prediction"]
    assert (
        answers.columns.tolist() == expected_answer_columns
    ), f"Unexcpected columns in answers, expected {expected_answer_columns}, got {answers.columns.tolist()}"
    assert answers["customer_id"].nunique() == len(
        answers
    ), "There should be no duplicate customer_ids in answers"
