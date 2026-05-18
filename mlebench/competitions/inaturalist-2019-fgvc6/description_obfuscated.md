# Task

Predict the species of plants and animals from images.

# Metric

Top-1 classification error. 

# Submission Format

For each image in the test set, you must predict 1 category label. However, you may predict more categories labels (sorted by confidence). The csv file should contain a header and have the following format:

```
id,predicted
268243,71 108 339 341 560
268244,333 729 838 418 785
268245,690 347 891 655 755
```

The `id` column corresponds to the test image id. The `predicted` column corresponds to 1 category id. The first category id will be used to compute the metric. You should have one row for each test image.

# Dataset

- **train_val2019.tar.gz** - Contains the training and validation images in a directory structure following {iconic category name}/{category name}/{image id}.jpg .
- **train2019.json** - Contains the training annotations.
- **val2019.json** - Contains the validation annotations.
- **test2019.tar.gz** - Contains a single directory of test images.
- **test2019.json** - Contains test image information.


We follow the annotation format of the COCO dataset and add additional fields.