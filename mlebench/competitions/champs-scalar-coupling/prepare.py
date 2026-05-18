import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    grouped_by_molecule = list(old_train.groupby("molecule_name"))
    train_groups, test_groups = train_test_split(grouped_by_molecule, test_size=0.1, random_state=0)
    new_train = pd.concat([group for _, group in train_groups])
    answers = pd.concat([group for _, group in test_groups])
    new_test = answers.drop(columns=["scalar_coupling_constant"])

    # Create sample submission
    sample_submission = new_test[["id"]].copy()
    sample_submission["scalar_coupling_constant"] = 0

    # Molecule structure data in CSV format
    structures = read_csv(raw / "structures.csv")
    structures = structures[structures["molecule_name"].isin(new_train["molecule_name"])]

    # Additional data CSVs
    dipole_moments = read_csv(raw / "dipole_moments.csv")
    dipole_moments = dipole_moments[
        dipole_moments["molecule_name"].isin(new_train["molecule_name"])
    ]

    magnetic_shielding_tensors = read_csv(raw / "magnetic_shielding_tensors.csv")
    magnetic_shielding_tensors = magnetic_shielding_tensors[
        magnetic_shielding_tensors["molecule_name"].isin(new_train["molecule_name"])
    ]

    mulliken_charges = read_csv(raw / "mulliken_charges.csv")
    mulliken_charges = mulliken_charges[
        mulliken_charges["molecule_name"].isin(new_train["molecule_name"])
    ]

    potential_energy = read_csv(raw / "potential_energy.csv")
    potential_energy = potential_energy[
        potential_energy["molecule_name"].isin(new_train["molecule_name"])
    ]

    scalar_coupling_contributions = read_csv(raw / "scalar_coupling_contributions.csv")
    scalar_coupling_contributions = scalar_coupling_contributions[
        scalar_coupling_contributions["molecule_name"].isin(new_train["molecule_name"])
    ]

    # Checks before writing
    data_csvs = {
        "structures": structures,
        "dipole_moments": dipole_moments,
        "magnetic_shielding_tensors": magnetic_shielding_tensors,
        "mulliken_charges": mulliken_charges,
        "potential_energy": potential_energy,
        "scalar_coupling_contributions": scalar_coupling_contributions,
    }
    for name, dataset in data_csvs.items():
        assert set(dataset["molecule_name"]) == set(
            new_train["molecule_name"]
        ), f"Filtered {name} should exactly match the molecule names present in the new_train set."

    assert set(new_train["molecule_name"]).isdisjoint(
        set(new_test["molecule_name"])
    ), "Train and test sets should not share any samples with the same molecule name."

    assert set(new_train["id"]).isdisjoint(
        set(new_test["id"])
    ), "Train and test sets should not share any samples with the same id."

    assert len(sample_submission) == len(
        new_test
    ), "Sample submission length does not match test length."

    assert (
        sample_submission.shape[1] == 2
    ), f"Sample submission should have 2 columns, but has {sample_submission.shape[1]}"

    assert new_test.shape[1] == 5, f"new_test should have 5 columns, but has {new_test.shape[1]}"

    assert answers.shape[1] == 6, f"answers should have 6 columns, but has {answers.shape[1]}"

    assert new_train.shape[1] == 6, f"new_train should have 6 columns, but has {new_train.shape[1]}"

    # Copy over molecule structure data individual files
    for molecule_name in tqdm(
        new_train["molecule_name"].unique(), desc="Copying molecule structure files"
    ):
        src_file = raw / "structures" / f"{molecule_name}.xyz"
        dst_file = public / "structures" / f"{molecule_name}.xyz"
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=src_file, dst=dst_file)

    # Write CSVs
    answers.to_csv(private / "answers.csv", index=False)

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    structures.to_csv(public / "structures.csv", index=False)
    dipole_moments.to_csv(public / "dipole_moments.csv", index=False)
    magnetic_shielding_tensors.to_csv(public / "magnetic_shielding_tensors.csv", index=False)
    mulliken_charges.to_csv(public / "mulliken_charges.csv", index=False)
    potential_energy.to_csv(public / "potential_energy.csv", index=False)
    scalar_coupling_contributions.to_csv(public / "scalar_coupling_contributions.csv", index=False)

    # Checks after writing
    assert len(list((public / "structures").glob("*.xyz"))) == len(
        new_train["molecule_name"].unique()
    ), "The number of files in public/structures should match the number of unique molecule names in the train set."
