# Task

Locate and classify each kuzushiji character on a page. While complete bounding boxes are provided for the training set, only a single point within the ground truth bounding box is needed for submissions.

# Metric

A modified version of the F1 Score. To score a true positive, you must provide center point coordinates that are within the ground truth bounding box and a matching label. The ground truth bounding boxes are defined in the format `{label X Y Width Height}`, so if the ground truth label is `U+003F 1 1 10 10` then a prediction of `U+003F 3 3` would pass.

# Submission Format

For each image in the test set, you must locate and identify all of the kuzushiji characters. Not every image will necessarily be scored. The file should contain a header and have the following format:

```
image_id,labels
image_id,{label X Y} {...}
```

Do not make more than 1,200 predictions per page.

# Dataset

- Some images only contain illustrations, in which case no labels should be submitted for the page.

- Kuzushiji text is written such that annotations are placed between the columns of the main text, usually in a slightly smaller font. The annotation characters should be ignored; labels for them will be scored as false positives.

- You can occasionally see through especially thin paper and read characters from the opposite side of the page. Those characters should also be ignored.

- **train.csv** - the training set labels and bounding boxes.
    - `image_id`: the id code for the image.
    - `labels`: a string of all labels for the image. The string should be read as space separated series of values where `Unicode character`, `X`, `Y`, `Width`, and `Height` are repeated as many times as necessary.
- **sample_submission.csv** - a sample submission file in the correct format.
    - `image_id`: the id code for the image
    - `labels`: a string of all labels for the image. The string should be read as space separated series of values where\
        `Unicode character`, `X`, and `Y` are repeated as many times as necessary. The default label predicts that there are the same two characters on every page, centered at pixels `(1,1)` and `(2,2)`.
- **unicode_translation.csv** - supplemental file mapping between unicode IDs and Japanese characters.
- **[train/test]_images.zip** - the images.