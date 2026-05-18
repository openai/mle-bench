import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mlebench.competitions.utils import rles_to_masks
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import dice_coefficient


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    id_col = "id"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing the '{id_col}' column")
    if "predicted" not in submission.columns:
        raise InvalidSubmissionError("Submission is missing the 'predicted' column")

    assert id_col in answers.columns, f"Answers is missing the '{id_col}' column"
    assert "encoding" in answers.columns, "Answers is missing the 'encoded' column"
    assert "width_pixels" in answers.columns, "Answers is missing the 'width_pixels' column"
    assert "height_pixels" in answers.columns, "Answers is missing the 'height_pixels' column"

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows while answers has {len(answers)} rows"
        )

    if set(submission[id_col]) != set(answers[id_col]):
        raise InvalidSubmissionError(f"Submission and answers have different {id_col} values")

    submission = submission.sort_values(id_col).reset_index(drop=True)
    answers = answers.sort_values(id_col).reset_index(drop=True)

    # pandas reads empty cells as nan, we mark them as empty RLE strings
    submission["predicted"] = submission["predicted"].fillna("")
    submission["answers"] = submission["predicted"].fillna("")

    image_heights = answers["height_pixels"].tolist()
    image_widths = answers["width_pixels"].tolist()

    prediced_masks = rles_to_masks(submission["predicted"].to_list(), image_heights, image_widths)
    true_masks = rles_to_masks(answers["encoding"], image_heights, image_widths)

    return prediced_masks, true_masks


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Computes the mean dice coefficient for the submission and answers.
    """
    predicted_masks, true_masks = prepare_for_metric(submission, answers)

    return np.mean(
        [
            dice_coefficient(predicted_mask, true_mask, both_empty_value=1.0)
            for predicted_mask, true_mask in tqdm(
                zip(predicted_masks, true_masks), total=len(predicted_masks)
            )
        ]
    )
