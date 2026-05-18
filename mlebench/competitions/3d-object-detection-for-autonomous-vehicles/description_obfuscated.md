# Task

Build a 3D object detection model.


# Metric

Mean average precision at different intersection over union (IoU) thresholds. The IoU of a set of predicted 3D bounding volumes and ground truth bounding volumes is calculated as:

$\operatorname{IoU}(A, B)=\frac{A \cap B}{A \cup B}.$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. At a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}.$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object.

If there are no ground truth objects at all for a given image, ANY number of predictions (false positives) will result in the image receiving a score of zero, and being included in the mean average precision.

The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold:

$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}.$

In your submission, you are also asked to provide a `confidence` level for each bounding box. Bounding boxes will be evaluated in order of their confidence levels in the above process. This means that bounding boxes with higher confidence will be checked first for matches against solutions, which determines what boxes are considered true and false positives.

If you do not wish to use or calculate `confidence` you can use a placeholder value - like `1.0` - to indicate that no particular order applies to the evaluation of your submission boxes.

The difference between the 2D and 3D bounding volume contexts is small. In the 3D context we reduce the bounding volume to a `ground bounding box` and a `height`. The IoU is then the `intersection` of the `ground bounding boxes` * the `intersection` of the `height` differences, divided by the union of the bounding boxes.

# Submission Format

The submission format requires a space delimited set of bounding volume parameters. For example:

`97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c,1.0 2742.15 673.16 -18.65 1.834 4.609 1.648 2.619 car`

indicates that sample `97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c` has a bounding volume with a `confidence` of 0.5, `center_x` of 2742.15, `center_y` of 673.16, `center_z` of -18.65, `width` of 1.834, `length` of 4.609, `height` of 1.648, `yaw` of 2.619, and a `class_name` of `car`.

The file should contain a header and have the following format. Each row in your submission should contain ALL bounding boxes for a given image.

```
Id,PredictionString
db8b47bd4ebdf3b3fb21598bb41bd8853d12f8d2ef25ce76edd4af4d04e49341,
97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c,1.0 2742.15 673.16 -18.65 1.834 4.609 1.648 2.619 car
etc...
```

# Dataset

The data comes in the form of many interlocking tables and formats. The JSON files all contain single tables with identifying `tokens` that can be used to join with other files / tables. The images and lidar files all correspond to a sample in `sample_data.json`, and the `sample_token` from `sample_data.json` is the primary identifier used for the train and test samples.

The annotations in `train.csv` are in the following format:\
`center_x center_y center_z width length height yaw class_name`

-   `center_x`, `center_y` and `center_z` are the world coordinates of the center of the 3D bounding volume.
-   `width`, `length` and `height` are the dimensions of the volume.
-   `yaw` is the angle of the volume around the `z` axis (where `y` is forward/back, `x` is left/right, and `z` is up/down - making 'yaw' the direction the front of the vehicle / bounding box is pointing at while on the ground).
-   `class_name` is the type of object contained by the bounding volume.

-   **train_data.zip** and **test_data.zip** - contains JSON files with multiple tables. The most important is `sample_data.json`, which contains the primary identifiers used in the competition, as well as links to key image / lidar information.
-   **train_images.zip** and **test_images.zip** - contains .jpeg files corresponding to samples in `sample_data.json`
-   **train_lidar.zip** and **test_lidar.zip** - contains .jpeg files corresponding to samples in `sample_data.json`
-   **train_maps.zip** and **test_maps.zip** - contains maps of the entire sample area.
-   **train.csv** - contains all `sample_token`s in the train set, as well as annotations in the required format for all train set objects.
-   **sample_submission.csv** - contains all `sample_token`s in the test set, with empty predictions.