import multiprocessing
from functools import partial
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from tqdm import tqdm

from mlebench.grade_helpers import InvalidSubmissionError

from .mAP_evaluation import get_average_precisions, get_class_names


def parse_row(sample_token: str, prediction_string: str) -> List[Dict[str, Any]]:
    """
    Parse a row comprising of a sample_token and a prediction_string

    `prediction_string` has the form:
        `{confidence} center_x center_y center_z width length height yaw class_name`
        where `confidence` is optional

        Example row:
        ```
        97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c,1.0 2742.152625996093 673.1631800662494 -18.6561112411676 1.834 4.609 1.648 2.619835541569646 car 0.5 2728.9634555684484 657.8296521874645 -18.54676216218047 1.799 4.348 1.728 -0.5425527100619654 bus
        ```

    Returns list of dicts with keys "sample_token", "translation", "size", "rotation", "name", "score", e.g.
    ```
    [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]
    ```
    """
    if prediction_string.strip() == "":
        return []
    prediction_string = prediction_string.split(" ")

    single_prediction_length = 9
    if len(prediction_string) % single_prediction_length != 0:
        raise InvalidSubmissionError(
            f"Prediction string must have {single_prediction_length} tokens per object, but got {len(prediction_string)} tokens"
        )

    n_predictions = len(prediction_string) / single_prediction_length
    preds = []
    for i in range(int(n_predictions)):
        prediction_segment = prediction_string[
            i * single_prediction_length : (i + 1) * single_prediction_length
        ]
        confidence = float(prediction_segment[0])
        prediction_segment = prediction_segment[1:]

        # Parse the prediction string
        translation = [float(x) for x in prediction_segment[0:3]]
        size = [float(x) for x in prediction_segment[3:6]]
        yaw = float(prediction_segment[6])
        name = prediction_segment[7]

        quat = Quaternion(axis=[0, 0, 1], radians=yaw)
        # (from https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/discussion/110884#638390)

        preds.append(
            {
                "sample_token": sample_token,
                "translation": translation,
                "size": size,
                "rotation": list(quat),
                "name": name,
                "score": confidence,
            }
        )
    return preds


def process_sample(gt_pred_pair, iou_thresholds):
    gt, predictions = gt_pred_pair
    class_names = get_class_names(gt)
    sample_mAPs = []
    for iou_threshold in iou_thresholds:
        average_precisions = get_average_precisions(gt, predictions, class_names, iou_threshold)
        mAP = np.mean(average_precisions)
        sample_mAPs.append(mAP)
    return sample_mAPs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    submission and answers are dataframes with columns `Id` and `PredictionString`

    We grade the submission using the 3D mAP metric described in the competition overview:
    https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/overview/evaluation

    The competition organizers provide a Python script (mAP_evaluation.py) to perform this metric calculation
    (search text for "3d version of the COCO mAP") for a (gt, predictions) pair from a single sample.
    https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/discussion/133895

    For each sample, we first parse the PredictionString in the format expected by mAP_evaluation.py, then calculate
    the mAP over the requested range of IoU thresholds, and get a mean mAP over all thresholds for that sample.

    The final score is the mean over all samples (i.e. the mean of the mean mAPs).
    """
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    if "Id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have an 'Id' column")

    if "PredictionString" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'PredictionString' column")

    assert "Id" in answers.columns, "Answers must have an 'Id' column"
    assert "PredictionString" in answers.columns, "Answers must have a 'PredictionString' column"

    submission = submission.sort_values("Id")
    answers = answers.sort_values("Id")

    if (submission["Id"].values != answers["Id"].values).any():
        raise InvalidSubmissionError("Submission and answers must have the same ids")

    # Empty values in the PredictionString column are allowed, but must be filled with an empty string
    # (pandas converts them to NaN, so we convert them back)
    submission["PredictionString"] = submission["PredictionString"].fillna("")

    # Parse each row into the format expected by mAP_evaluation.py
    submission_samples = [
        parse_row(row["Id"], row["PredictionString"]) for _, row in submission.iterrows()
    ]
    answer_samples = [
        parse_row(row["Id"], row["PredictionString"]) for _, row in answers.iterrows()
    ]

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # (from https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/overview/evaluation)

    # Prepare the data for parallel processing
    sample_pairs = list(zip(answer_samples, submission_samples))

    # Use multiprocessing to parallelize the computation
    num_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(partial(process_sample, iou_thresholds=iou_thresholds), sample_pairs),
                total=len(sample_pairs),
                desc="Processing samples",
            )
        )

    # Flatten the results
    mAPs = [mAP for sample_mAPs in results for mAP in sample_mAPs]

    # Average over all samples and IoU thresholds
    final_mAP = np.mean(mAPs)

    return final_mAP
