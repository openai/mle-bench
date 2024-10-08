# Task

Build a model to classify the species of plants from images.

# Metric

Macro F1 score. A separate F1 score is calculated for each `species` value and then averaged.

# Submission Format

For each image `Id`, you should predict the corresponding image label ("category_id") in the `Predicted` column. The submission file should have the following format:

```
Id,Predicted
0,0
1,27
2,42
...
```

# Dataset 

This dataset uses the [COCO dataset format](http://cocodataset.org/#format-data) with additional annotation fields. In addition to the species category labels, we also provide region and supercategory information.

The training set images are organized in subfolders `train/<subfolder1>/<subfolder2>/<image id>.jpg`.

The test set images are organized in subfolders `test/<subfolder>/<image id>.jpg`.