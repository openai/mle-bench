import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """Prepare the submission and answers for the metric."""

    required_answer_columns = ["class_id", "x_min", "y_min", "x_max", "y_max", "image_id"]
    for col in required_answer_columns:
        assert col in answers.columns, f"Answers must have a `{col}` column"
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        assert (
            not answers[col].isnull().values.any()
        ), f"Answers must not contain any NaN values in `{col}` column, but got {answers[col].isnull().values}"

    required_submission_columns = ["PredictionString", "image_id"]
    for col in required_submission_columns:
        if col not in submission.columns:
            raise InvalidSubmissionError(f"Submission must have a `{col}` column")

    if set(submission["image_id"]) != set(answers["image_id"]):
        raise InvalidSubmissionError(
            "The set of image_ids in the submission must match the set of image_ids in the answers"
        )

    return {"true_df": answers, "pred_df": submission}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    inputs = prepare_for_metric(submission, answers)
    vineval = VinBigDataEval(inputs["true_df"])
    cocoEvalResults = vineval.evaluate(inputs["pred_df"])
    return cocoEvalResults.stats[0]


class VinBigDataEval:
    """Helper class for calculating the competition metric.

    You should remove the duplicated annoatations from the `true_df` dataframe
    before using this script. Otherwise it may give incorrect results.

        >>> vineval = VinBigDataEval(valid_df)
        >>> cocoEvalResults = vineval.evaluate(pred_df)

    Arguments:
        true_df: pd.DataFrame Clean (no duplication) Training/Validating dataframe.

    Authors:
        Peter (https://kaggle.com/pestipeti)

    See:
        https://www.kaggle.com/pestipeti/competition-metric-map-0-4

    Returns: None

    """

    def __init__(self, true_df):

        self.true_df = true_df

        self.image_ids = true_df["image_id"].unique()
        self.annotations = {
            "type": "instances",
            "images": self.__gen_images(self.image_ids),
            "categories": self.__gen_categories(self.true_df),
            "annotations": self.__gen_annotations(self.true_df, self.image_ids),
        }

        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None,
        }

    def __gen_categories(self, df):
        print("Generating category data...")

        if "class_name" not in df.columns:
            df["class_name"] = df["class_id"]

        cats = df[["class_name", "class_id"]]
        cats = cats.drop_duplicates().sort_values(by="class_id").values

        results = []

        for cat in cats:
            results.append(
                {
                    "id": cat[1],
                    "name": cat[0],
                    "supercategory": "none",
                }
            )

        return results

    def __gen_images(self, image_ids):
        print("Generating image data...")
        results = []

        for idx, image_id in enumerate(image_ids):

            # Add image identification.
            results.append(
                {
                    "id": idx,
                }
            )

        return results

    def __gen_annotations(self, df, image_ids):
        print("Generating annotation data...")
        k = 0
        results = []

        for idx, image_id in enumerate(image_ids):

            # Add image annotations
            for i, row in df[df["image_id"] == image_id].iterrows():

                results.append(
                    {
                        "id": k,
                        "image_id": idx,
                        "category_id": row["class_id"],
                        "bbox": np.array([row["x_min"], row["y_min"], row["x_max"], row["y_max"]]),
                        "segmentation": [],
                        "ignore": 0,
                        "area": (row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"]),
                        "iscrowd": 0,
                    }
                )

                k += 1

        return results

    def __decode_prediction_string(self, pred_str):
        data = list(map(float, pred_str.split(" ")))
        data = np.array(data)

        return data.reshape(-1, 6)

    def __gen_predictions(self, df, image_ids):
        print("Generating prediction data...")
        k = 0
        results = []

        for i, row in df.iterrows():

            image_id = row["image_id"]
            preds = self.__decode_prediction_string(row["PredictionString"])

            for j, pred in enumerate(preds):

                results.append(
                    {
                        "id": k,
                        "image_id": int(np.where(image_ids == image_id)[0]),
                        "category_id": int(pred[0]),
                        "bbox": np.array([pred[2], pred[3], pred[4], pred[5]]),
                        "segmentation": [],
                        "ignore": 0,
                        "area": (pred[4] - pred[2]) * (pred[5] - pred[3]),
                        "iscrowd": 0,
                        "score": pred[1],
                    }
                )

                k += 1

        return results

    def evaluate(self, pred_df, n_imgs=-1):
        """Evaluating your results

        Arguments:
            pred_df: pd.DataFrame your predicted results in the
                     competition output format.

            n_imgs:  int Number of images use for calculating the
                     result.All of the images if `n_imgs` <= 0

        Returns:
            COCOEval object
        """

        if pred_df is not None:
            self.predictions["annotations"] = self.__gen_predictions(pred_df, self.image_ids)

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()

        imgIds = sorted(coco_ds.getImgIds())

        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, "bbox")
        cocoEval.params.imgIds = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([0.4])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval
