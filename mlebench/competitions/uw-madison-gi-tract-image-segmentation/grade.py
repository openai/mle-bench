import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import directed_hausdorff
from tqdm.auto import tqdm

from mlebench.competitions.utils import rles_to_masks
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import dice_coefficient

UNIT_CUBE_DIAGONAL = np.sqrt(3)


def group_masks_by_day(masks: list[np.ndarray], cases_days_slices: list[str]) -> list[np.ndarray]:
    """
    Groups the 2D masks into 3D masks based on the case days
    - We first join masks of the same slice into a single mask with an OR operation
    - We then stack them with all the other slices from the same day in the same case into a 3D image
    """
    case_day_slice_to_masks = {}
    unique_slices = []  # list to keep ordering consistent
    for mask, case_day_slice in sorted(zip(masks, cases_days_slices), key=lambda x: x[1]):
        if case_day_slice not in case_day_slice_to_masks:
            case_day_slice_to_masks[case_day_slice] = []
            unique_slices.append(case_day_slice)
        case_day_slice_to_masks[case_day_slice].append(mask)

    case_day_to_masks = {}
    case_days = []  # list to keep ordering consistent
    for case_day_slice in unique_slices:
        # go from e.g. "case123_day2_slice_0001" to "case123_day2"
        case_day = case_day_slice.split("_slice_")[0]
        if case_day not in case_day_to_masks:
            case_day_to_masks[case_day] = []
            case_days.append(case_day)
        # OR on the masks from the same case_day_slice
        joined_mask = np.logical_or.reduce(case_day_slice_to_masks[case_day_slice]).astype(np.uint8)
        # add the joined slice to its relevant case_day
        case_day_to_masks[case_day].append(joined_mask)

    # now we stack the joined masks for each case_day to make a 3d image
    for case_day in case_day_to_masks:
        # make into 3d image
        case_day_to_masks[case_day] = np.stack(case_day_to_masks[case_day], axis=0)

    # keep the order of the cases consistent
    case_day_masks = [case_day_to_masks[case_day] for case_day in case_days]
    return case_day_masks


def prepare_for_metric(submission: DataFrame, answers: DataFrame):
    """
    Answers dataframe contains image_height and image_width columns necessary for RLE decoding
    """

    assert "image_height" in answers.columns, "image_height column not found in answers"
    assert "image_width" in answers.columns, "image_width column not found in answers"
    if not {"id", "class", "predicted"}.issubset(set(submission.columns)):
        raise InvalidSubmissionError("Submission must have columns: id, class, predicted")

    # prepare appropriate types
    submission["predicted"] = submission["predicted"].fillna("")
    answers["predicted"] = answers["predicted"].fillna("")
    answers["image_height"] = answers["image_height"].astype(int)
    answers["image_width"] = answers["image_width"].astype(int)

    image_heights = answers["image_height"].tolist()
    image_widths = answers["image_width"].tolist()

    # extract masked images from RLE in the dataframes
    submission_masks = rles_to_masks(submission["predicted"].to_list(), image_heights, image_widths)
    answer_masks = rles_to_masks(answers["predicted"].to_list(), image_heights, image_widths)

    # group 2d slice masks into 3d case_day masks
    submission_masks_by_day = group_masks_by_day(submission_masks, submission["id"])
    answer_masks_by_day = group_masks_by_day(answer_masks, answers["id"])

    return submission_masks, answer_masks, submission_masks_by_day, answer_masks_by_day


def hausdorff_distance(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Computes the hausdorff distance between two 3-dimensional binary masks (H, W, C)

    Args:
        predicted_mask: A 3D binary numpy array indicating where the segmentation is predicted
        true_mask: A 3D binary numpy array indicating where the segmentation is
    """
    # if both empty, return nan, if identical return 0, if only one is empty, return 1
    if np.sum(predicted_mask) == 0 and np.sum(true_mask) == 0:
        return np.nan
    elif np.array_equal(predicted_mask, true_mask):
        return 0
    elif (np.sum(predicted_mask) == 0 and np.sum(true_mask) > 0) or (
        np.sum(predicted_mask) > 0 and np.sum(true_mask) == 0
    ):
        return 1

    # if the pred mask is > 1 OOM larger than the true mask, return 1 to avoid unnecess heavy computation
    if predicted_mask.sum() > 10 * true_mask.sum():
        return 1

    true_coordinates = np.argwhere(true_mask)
    predicted_coordinates = np.argwhere(predicted_mask)

    # normalize so that the coordinates are in the range [0, 1], to get a unit cube
    normalization_constants = true_mask.shape
    true_coordinates = true_coordinates / normalization_constants
    predicted_coordinates = predicted_coordinates / normalization_constants

    # Symmetrize Hausdorff distance
    hausdorff_distance1, _, _ = directed_hausdorff(true_coordinates, predicted_coordinates)
    hausdorff_distance2, _, _ = directed_hausdorff(predicted_coordinates, true_coordinates)
    hausdorff_distance = max(hausdorff_distance1, hausdorff_distance2)

    # normalize by the maximum distance possible in a unit cube, which is the diagonal
    hausdorff_distance = hausdorff_distance / UNIT_CUBE_DIAGONAL

    return hausdorff_distance


def grade(submission: DataFrame, answers: DataFrame) -> float:
    """
    Computes dice_weight * dice + hausdorff_weight * (1 - hausdorff)
    """
    dice_weight = 0.4
    hausdorff_weight = 0.6

    (
        submission_masks,
        answer_masks,
        submission_masks_by_day,
        answer_masks_by_day,
    ) = prepare_for_metric(submission, answers)

    # we skip cases where both masks are empty, so we use nanmean
    # https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/324934#1795540
    dice_coeff = np.nanmean(
        [
            dice_coefficient(submission_mask, answer_mask, np.nan)
            for submission_mask, answer_mask in tqdm(
                zip(submission_masks, answer_masks),
                total=len(submission_masks),
            )
        ]
    )

    # skip cases where both masks are empty, to match behaviour of dice
    hausdorff_dis = np.nanmean(
        [
            hausdorff_distance(submission_mask_by_day, answer_mask_by_day)
            for submission_mask_by_day, answer_mask_by_day in tqdm(
                zip(submission_masks_by_day, answer_masks_by_day),
                total=len(submission_masks_by_day),
            )
        ]
    )

    return dice_weight * dice_coeff + hausdorff_weight * (1 - hausdorff_dis)
