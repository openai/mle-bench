from typing import Tuple

from pandas import DataFrame
from sklearn.metrics import roc_auc_score

from mlebench.grade_helpers import InvalidSubmissionError


def _filter_for_subgroup(
    answers: DataFrame, submission: DataFrame, identity_column: str, target_value: int = 1
) -> Tuple[DataFrame, DataFrame]:
    """
    Restricts the data to only the examples that mention the specific identity subgroup.
    A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity.
    """

    filtered_answers = answers[answers[identity_column] == target_value]
    filtered_submission = submission[submission["id"].isin(filtered_answers["id"])]
    return filtered_answers, filtered_submission


def _filter_for_bpsn(
    answers: DataFrame, submission: DataFrame, identity_column: str, target_value: int = 1
) -> Tuple[DataFrame, DataFrame]:
    """
    Filter for BPSN (Background Positive, Subgroup Negative) AUC
    Restricts the data to only the non-toxic examples that mention the identity and the toxic examples that do not.
    A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.
    """
    filtered_answers = answers[
        ((answers["toxicity"] == 0) & (answers[identity_column] == target_value))
        | ((answers["toxicity"] == 1) & (answers[identity_column] != target_value))
    ]
    filtered_submission = submission[submission["id"].isin(filtered_answers["id"])]
    return filtered_answers, filtered_submission


def _filter_for_bnsp(
    answers: DataFrame, submission: DataFrame, identity_column: str, target_value: int = 1
) -> Tuple[DataFrame, DataFrame]:
    """
    Filter for BNSP (Background Negative, Subgroup Positive) AUC
    Restricts the data to only the toxic examples that mention the identity and the non-toxic examples that do not.
    A low value in this metric means that the model confuses toxic examples that mention the identity with non-toxic examples that do not, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.
    """
    filtered_answers = answers[
        ((answers["toxicity"] == 1) & (answers[identity_column] == target_value))
        | ((answers["toxicity"] == 0) & (answers[identity_column] != target_value))
    ]
    filtered_submission = submission[submission["id"].isin(filtered_answers["id"])]
    return filtered_answers, filtered_submission


def _generalized_mean(scores: list, p: int, identity_columns: list[str]) -> float:
    return (sum([s**p for s in scores]) / len(identity_columns)) ** (1 / p)


def prepare_for_metric(
    submission: DataFrame, answers: DataFrame, identity_columns: list[str]
) -> Tuple[DataFrame, DataFrame]:

    # Ensure that the submission DataFrame contains the necessary columns
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission DataFrame must have an 'id' column.")
    if (
        "prediction" not in submission.columns and "toxicity" not in submission.columns
    ):  # Allow for the submission to be the CSV of answers, which has a "toxicity" column instead of a "prediction" column
        raise InvalidSubmissionError("Submission DataFrame must have a 'prediction' column.")

    # Ensure that the answers DataFrame contains the necessary columns
    assert "id" in answers.columns, "Answers DataFrame must have an 'id' column."
    assert "toxicity" in answers.columns, "Answers DataFrame must have a 'toxicity' column."

    submission = submission.sort_values("id")
    answers = answers.sort_values("id")

    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("IDs in the submission and answers DataFrames do not match.")

    # Ensure that all identity columns are present in the answers DataFrame
    for identity_column in identity_columns:
        assert (
            identity_column in answers.columns
        ), f"Answers DataFrame must have an '{identity_column}' column."

    submission.rename(columns={"prediction": "toxicity"}, inplace=True)

    # Convert continuous toxicity scores to binary toxicity labels
    submission["toxicity"] = submission["toxicity"].apply(lambda x: 1 if x >= 0.5 else 0)
    answers["toxicity"] = answers["toxicity"].apply(lambda x: 1 if x >= 0.5 else 0)

    # Convert continuous identity columns to binary identity columns
    for identity_column in identity_columns:
        answers[identity_column] = answers[identity_column].apply(lambda x: 1 if x >= 0.5 else 0)

    return submission, answers


def grade(
    submission: DataFrame,
    answers: DataFrame,
    identity_columns: list[str] = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ],
    p: int = -5,
    w: float = 0.25,
) -> float:
    """
    Computes the custom metric for Jigsaw Unintended Bias in Toxicity Classification.
    See https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation for details on this metric.
    """

    submission, answers = prepare_for_metric(submission, answers, identity_columns)

    overall_auc = roc_auc_score(answers["toxicity"], submission["toxicity"])

    # To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias.
    bias_aucs = []
    for identity_column in identity_columns:
        subgroup_answers, subgroup_submission = _filter_for_subgroup(
            answers, submission, identity_column
        )
        bpsn_answers, bpsn_submission = _filter_for_bpsn(answers, submission, identity_column)
        bnsp_answers, bnsp_submission = _filter_for_bnsp(answers, submission, identity_column)

        subgroup_aucs = roc_auc_score(subgroup_answers["toxicity"], subgroup_submission["toxicity"])
        bpsn_aucs = roc_auc_score(bpsn_answers["toxicity"], bpsn_submission["toxicity"])
        bnsp_aucs = roc_auc_score(bnsp_answers["toxicity"], bnsp_submission["toxicity"])

        bias_aucs.append([subgroup_aucs, bpsn_aucs, bnsp_aucs])

    # Calculate generalized mean for each identity column
    generalized_means = [
        _generalized_mean([bias_auc[i] for bias_auc in bias_aucs], p, identity_columns)
        for i in range(3)  # there are 3 bias metrics
    ]

    # Combine the overall AUC with the generalized mean of the Bias AUCs
    weighted_overall_auc = w * overall_auc
    weighted_generalized_means = sum(
        [w * generalized_means[i] for i in range(len(generalized_means))]
    )
    final_score = weighted_overall_auc + weighted_generalized_means

    return final_score
