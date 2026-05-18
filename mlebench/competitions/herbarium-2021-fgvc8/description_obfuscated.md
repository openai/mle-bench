# Task

Build a model to classify the species of plants from images.

# Metric

Macro F1 score. A separate F1 score is calculated for each `species` value and then averaged.

# Submission Format

For each image `Id`, you should predict the corresponding image label (`category_id`) in the `Predicted` column. The submission file should have the following format:

```
Id,Predicted
0,1
1,27
2,42
...
```

# Dataset

This dataset uses the [COCO dataset format](https://cocodataset.org/#format-data) with additional annotation fields. In addition to the species category labels, we also provide region and supercategory information.

The training set images are organized in subfolders `train/images/<subfolder1>/<subfolder2>/<image id>.jpg`, where `<subfolder1>` combined with `<subfolder2>` corresponds to the `category_id`. For example, a training image with an `image_id` of `1104517` and a `category_id` of `00001`, can be found at `train/images/000/01/1104517.jpg`.

The test set images are organized in subfolders `test/images/<subfolder>/<image id>.jpg`, where `<subfolder>` corresponds to the integer division of the `image_id` by 1000. For example, a test image with and `image_id` of `8005`, can be found at `test/images/008/8005.jpg`.