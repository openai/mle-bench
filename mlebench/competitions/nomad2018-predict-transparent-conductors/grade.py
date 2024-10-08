from typing import Tuple

import pandas as pd
from sklearn import metrics

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    assert (
        "formation_energy_ev_natom" in answers.columns
    ), "Expected 'formation_energy_ev_natom' column in answers"
    assert "bandgap_energy_ev" in answers.columns, "Expected 'bandgap_energy_ev' column in answers"
    if "formation_energy_ev_natom" not in submission.columns:
        raise InvalidSubmissionError("Expected 'formation_energy_ev_natom' column in submission")
    if "bandgap_energy_ev" not in submission.columns:
        raise InvalidSubmissionError("Expected 'bandgap_energy_ev' column in submission")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected {len(answers)} rows in submission, got {len(submission)}"
        )

    true_labels_formation, true_labels_bandgap = (
        answers["formation_energy_ev_natom"],
        answers["bandgap_energy_ev"],
    )
    predictions_formation, predictions_bandgap = (
        submission["formation_energy_ev_natom"],
        submission["bandgap_energy_ev"],
    )

    return true_labels_formation, true_labels_bandgap, predictions_formation, predictions_bandgap


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    (
        true_labels_formation,
        true_labels_bandgap,
        predictions_formation,
        predictions_bandgap,
    ) = prepare_for_metric(submission, answers)
    return (
        metrics.root_mean_squared_log_error(true_labels_formation, predictions_formation)
        + metrics.root_mean_squared_log_error(true_labels_bandgap, predictions_bandgap)
    ) / 2
