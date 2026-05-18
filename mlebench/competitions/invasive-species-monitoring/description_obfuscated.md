# Task

Predict the presence of an invasive hydrangea species in images of foliage.

# Metric

Area under the ROC curve.

# Submission Format

For each image in the test set, you must predict a probability for the target variable on whether the image contains invasive species or not. The file should contain a header and have the following format:

```
name,invasive
2,0.5
5,0
6,0.2
etc.
```

# Dataset

The data set contains pictures taken in a forest.

- **train.7z** - the training set (contains 2295 images).
- **train_labels.csv** - the correct labels for the training set.
- **test.7z** - the testing set (contains 1531 images), ready to be labeled by your algorithm.
- **sample_submission.csv** - a sample submission file in the correct format.

## Data fields

- **name** - name of the sample picture file (numbers)
- **invasive** - probability of the picture containing an invasive species. A probability of 1 means the species is present.