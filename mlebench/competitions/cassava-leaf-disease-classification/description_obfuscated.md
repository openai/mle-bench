# Task

Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

# Metric

Categorization accuracy.

# Submission Format

```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

# Dataset

**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.
