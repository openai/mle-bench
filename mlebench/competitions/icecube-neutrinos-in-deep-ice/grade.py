import numpy as np
import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    # submission
    if set(submission.columns) != {"azimuth", "zenith", "event_id"}:
        raise InvalidSubmissionError(
            "Submission must contain columns 'azimuth','zenith' and 'event_id'"
        )

    if not np.issubdtype(submission["azimuth"].dtype, np.number):
        raise InvalidSubmissionError("Azimuth must be a number")
    if not np.issubdtype(submission["zenith"].dtype, np.number):
        raise InvalidSubmissionError("Zenith must be a number")
    if not np.all(np.isfinite(submission["azimuth"])):
        raise InvalidSubmissionError("Azimuth must not be infinite")
    if not np.all(np.isfinite(submission["zenith"])):
        raise InvalidSubmissionError("Zenith must not be infinite")
    if submission["azimuth"].isnull().any():
        raise InvalidSubmissionError("Azimuth must not be NaN")
    if submission["zenith"].isnull().any():
        raise InvalidSubmissionError("Zenith must not be NaN")

    # answers
    assert set(answers.columns) == {
        "azimuth",
        "zenith",
        "event_id",
    }, "Answers must contain columns 'azimuth','zenith' and 'event_id'"
    assert np.issubdtype(answers["azimuth"].dtype, np.number), "Azimuth must be a number"
    assert np.issubdtype(answers["zenith"].dtype, np.number), "Zenith must be a number"
    assert np.all(np.isfinite(answers["azimuth"])), "Azimuth must not be infinite"
    assert np.all(np.isfinite(answers["zenith"])), "Zenith must not be infinite"
    assert not answers["azimuth"].isnull().any(), "Azimuth must not be NaN"
    assert not answers["zenith"].isnull().any(), "Zenith must not be NaN"

    # both
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")
    if set(submission["event_id"]) != set(answers["event_id"]):
        raise InvalidSubmissionError("Submission and answers must have the same event_ids")

    # sort values by id so that the order is correct
    submission = submission.sort_values("event_id")
    answers = answers.sort_values("event_id")

    return {
        "az_true": answers["azimuth"].to_numpy(),
        "zen_true": answers["zenith"].to_numpy(),
        "az_pred": submission["azimuth"].to_numpy(),
        "zen_pred": submission["zenith"].to_numpy(),
    }


# courtesy of notebook from competition host: https://www.kaggle.com/code/sohier/mean-angular-error
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    """

    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    ang_dist_score_inputs = prepare_for_metric(submission, answers)
    score = angular_dist_score(**ang_dist_score_inputs)
    return score
