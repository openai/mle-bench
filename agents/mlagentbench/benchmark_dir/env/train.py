from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/")
SUBMISSION_PATH = Path("submission/submission.csv")

# OPTIONAL: Any functions you want to create here

if __name__ == "__main__":

    # TODO: Your code here
    print(f"To get started, view the competition task and dataset at {DATA_DIR}")

    # TODO: Make actual predictions and save to submission file
    submission = pd.DataFrame()
    submission.to_csv(SUBMISSION_PATH, index=False)
