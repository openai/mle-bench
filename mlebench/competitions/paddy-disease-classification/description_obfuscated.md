# Task

Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

# Metric

Categorization accuracy.

# Submission Format

```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

# Dataset

**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.
