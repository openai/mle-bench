from pandas import DataFrame
from scipy.sparse import spmatrix
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

from mlebench.grade_helpers import InvalidSubmissionError


def binarize_sentences(y_true: list[set[str]], y_pred: list[set[str]]) -> tuple[spmatrix, spmatrix]:
    """
    Binarizes lists of sentence sets so we can pass them as inputs to scipy.jaccard_score
    """
    all_words = list(set.union(*y_true, *y_pred))

    mlb = MultiLabelBinarizer(classes=all_words, sparse_output=True)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    return y_true_bin, y_pred_bin


def prepare_for_grading(
    submission: DataFrame, answers: DataFrame, target_column: str
) -> dict[str, spmatrix]:
    """
    Extracts the sentences from the dataframe
    Splits them into sets of lowercased words
    Binarizes these sets so we can pass them as inputs to scipy.jaccard_score
    """

    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission should be the same length as the answers")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an `id` column")
    assert target_column in answers.columns, f"Answers must have a `{target_column}` column"
    assert "id" in answers.columns, "Answers must have an `id` column"

    submission = submission.sort_values("id")
    answers = answers.sort_values("id")

    if (submission["id"].values != answers["id"].values).any():
        raise InvalidSubmissionError("Submission does not contain same IDs as test")

    # pandas reads empty cells as NaNs, which are float. We fill with empty string to match type
    submission[target_column] = submission[target_column].fillna("")
    answers[target_column] = answers[target_column].fillna("")

    y_pred = submission[target_column].tolist()
    y_true = answers[target_column].tolist()

    # split into sets of words
    y_pred_sentence_sets = [set(sentence.lower().split()) for sentence in y_pred]
    y_true_sentence_sets = [set(sentence.lower().split()) for sentence in y_true]

    # binarize for jaccard_score
    y_true_bin, y_pred_bin = binarize_sentences(y_true_sentence_sets, y_pred_sentence_sets)

    return {"y_pred": y_pred_bin, "y_true": y_true_bin}


def grade(submission: DataFrame, answers: DataFrame) -> float:
    grading_input = prepare_for_grading(submission, answers, "PredictionString")

    # we use average=`samples` to match what the competition description suggests
    return jaccard_score(**grading_input, average="samples")
