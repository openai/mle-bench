from pathlib import Path

import pandas as pd


def prepare(raw: Path, public: Path, private: Path) -> None:
    old_train = pd.read_json(raw / "train.json", lines=True)
    old_test = pd.read_json(raw / "test.json", lines=True)
    old_sample_submission = pd.read_csv(raw / "sample_submission.csv")

    to_predict = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    test_size = 0.1
    n_test_samples = int(len(old_train) * test_size)

    # only put samples that pass the SN filter in the test set, as per comp data desc
    old_train["test"] = False
    test_indices = (
        old_train[old_train["SN_filter"] > 0].sample(n=n_test_samples, random_state=0).index
    )
    old_train.loc[test_indices, "test"] = True

    new_train = old_train[~old_train["test"]].copy().drop(columns=["test"], inplace=False)
    new_test = old_train[old_train["test"]].copy().drop(columns=["test"], inplace=False)
    old_train = old_train.drop(columns=["test"], inplace=False)

    # Create `test.csv` by exploding each list in the `reactivity` and `deg_*` columns, analogous
    # to `pd.explode`. Only the first `seq_scored` items are scored out of a possible `seq_length`
    # items. For each row, we keep track of whether it's scored or not with the `keep` column.
    records = []

    for _, row in new_test.iterrows():
        n = row["seq_scored"]

        assert len(row["reactivity"]) == n
        assert len(row["deg_Mg_pH10"]) == n
        assert len(row["deg_pH10"]) == n
        assert len(row["deg_Mg_50C"]) == n
        assert len(row["deg_50C"]) == n

        for j in range(n):
            records.append(
                {
                    "id_seqpos": f"{row['id']}_{j}",
                    "reactivity": row["reactivity"][j],
                    "deg_Mg_pH10": row["deg_Mg_pH10"][j],
                    "deg_pH10": row["deg_pH10"][j],
                    "deg_Mg_50C": row["deg_Mg_50C"][j],
                    "deg_50C": row["deg_50C"][j],
                    "keep": True,
                }
            )

        k = row["seq_length"]

        assert n < k

        for j in range(n, k):
            records.append(
                {
                    "id_seqpos": f"{row['id']}_{j}",
                    "reactivity": 0.0,
                    "deg_Mg_pH10": 0.0,
                    "deg_pH10": 0.0,
                    "deg_Mg_50C": 0.0,
                    "deg_50C": 0.0,
                    "keep": False,
                }
            )

    # Write `answers.csv`
    answers = pd.DataFrame(records)
    answers.to_csv(private / "test.csv", index=False, float_format="%.10f")

    # Write `train.json`
    new_train["index"] = range(len(new_train))
    new_train.to_json(public / "train.json", orient="records", lines=True)

    # Write `test.json`
    new_test_without_labels = new_test[old_test.columns].copy()
    new_test_without_labels["index"] = range(len(new_test_without_labels))
    new_test_without_labels.to_json(public / "test.json", orient="records", lines=True)

    # Write `sample_submission.csv`
    new_sample_submission = answers[["id_seqpos"] + to_predict].copy()
    new_sample_submission.loc[:, to_predict] = 0.0
    new_sample_submission.to_csv(
        public / "sample_submission.csv", index=False, float_format="%.10f"
    )

    # Sanity checks
    assert set(new_train.columns) == set(old_train.columns), (
        f"Expected the columns of the new train to be the same as the old train, but got "
        f"{set(new_train.columns)} instead of {set(old_train.columns)}."
    )

    assert set(new_test_without_labels.columns) == set(old_test.columns), (
        f"Expected the columns of the new test to be the same as the old test, but got "
        f"{set(new_test_without_labels.columns)} instead of {set(old_test.columns)}."
    )

    assert set(to_predict).intersection(set(new_test_without_labels.columns)) == set(), (
        f"Expected the columns to predict aren't included in the new test, but got "
        f"{set(to_predict) ^ set(new_test_without_labels.columns)} instead of the empty set."
    )

    assert set(new_sample_submission.columns) == set(old_sample_submission.columns), (
        f"Expected the columns of the new sample submission to be the same as the old sample "
        f"submission, but got {set(new_sample_submission.columns)} instead of "
        f"{set(old_sample_submission.columns)}."
    )

    assert len(answers) == len(new_sample_submission), (
        f"Expected the answers to have the same length as the new sample submission, but got "
        f"{len(answers)} instead of {len(new_sample_submission)}."
    )

    # we can use [0] because all sequences have the same length
    assert len(new_sample_submission) == (
        len(new_test_without_labels) * new_test_without_labels["seq_length"].iloc[0]
    ), (
        "Expected new_sample_submission length to be equal to max seq_length * len(new_test)."
        f"Got {len(new_sample_submission)} instead of {len(new_test_without_labels) * new_test_without_labels['seq_length']}."
    )

    assert len(new_train) + len(new_test) == len(old_train), (
        f"Expected the length of the new train set plus the length of the new test set to be "
        f"equal to the length of the old train set, but got {len(new_train) + len(new_test)} "
        f"instead of {len(old_train)}."
    )
