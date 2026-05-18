# Overview

## Description

![autonomous-vehicle-lidar](https://storage.googleapis.com/kaggle-media/competitions/Lyft-Kaggle/Kaggle-01.png)

Self-driving technology presents a rare opportunity to improve the quality of life in many of our communities. Avoidable collisions, single-occupant commuters, and vehicle emissions are choking cities, while infrastructure strains under rapid urban growth. Autonomous vehicles are expected to redefine transportation and unlock a myriad of societal, environmental, and economic benefits. You can apply your data analysis skills in this competition to advance the state of self-driving technology.

[Lyft](https://level5.lyft.com/), whose mission is to improve people's lives with the world's best transportation, is investing in the future of self-driving vehicles. Level 5, their self-driving division, is working on a fleet of autonomous vehicles, and currently has a team of 450+ across Palo Alto, London, and Munich working to build a leading self-driving system ([they're hiring!](https://level5.lyft.com/#joinourteam)). Their goal is to democratize access to self-driving technology for hundreds of millions of Lyft passengers.

From a technical standpoint, however, the bar to unlock technical research and development on higher-level autonomy functions like perception, prediction, and planning is extremely high. This implies technical R&D on self-driving cars has traditionally been inaccessible to the broader research community.

This dataset aims to democratize access to such data, and foster innovation in higher-level autonomy functions for everyone, everywhere. By conducting a competition, we hope to encourage the research community to focus on hard problems in this space---namely, 3D object detection over semantic maps.

In this competition, you will build and optimize algorithms based on a large-scale dataset. This dataset features the raw sensor camera inputs as perceived by a fleet of multiple, high-end, autonomous vehicles in a restricted geographic area.

If successful, you'll make a significant contribution towards stimulating further development in autonomous vehicles and empowering communities around the world.

## Evaluation

This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a set of predicted 3D bounding volumes and ground truth bounding volumes is calculated as:

$\operatorname{IoU}(A, B)=\frac{A \cap B}{A \cup B}.$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}.$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object.

Important note: if there are no ground truth objects at all for a given image, ANY number of predictions (false positives) will result in the image receiving a score of zero, and being included in the mean average precision.

The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold:

$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}.$

In your submission, you are also asked to provide a `confidence` level for each bounding box. Bounding boxes will be evaluated in order of their confidence levels in the above process. This means that bounding boxes with higher confidence will be checked first for matches against solutions, which determines what boxes are considered true and false positives.

NOTE: In nearly all cases `confidence` will have no impact on scoring. It exists primarily to allow for submission boxes to be evaluated in a particular order to resolve extreme edge cases. None of these edge cases are known to exist in the data set. If you do not wish to use or calculate `confidence` you can use a placeholder value - like `1.0` - to indicate that no particular order applies to the evaluation of your submission boxes.

Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

### Intersection over Union (IoU)

Intersection over Union is a measure of the magnitude of overlap between two bounding boxes (or, in the more general case, two objects). It calculates the size of the overlap between two objects, divided by the total area of the two objects combined.

It can be visualized as the following:

![Image of Intersection over Union](https://storage.googleapis.com/kaggle-media/competitions/rsna/IoU.jpg)

The two boxes in the visualization overlap, but the area of the overlap is insubstantial compared with the area taken up by both objects together. IoU would be low - and would likely not count as a "hit" at higher IoU thresholds.

### 3D Context

The difference between the 2D and 3D bounding volume contexts is small. In the 3D context we reduce the bounding volume to a `ground bounding box` and a `height`. The IoU is then the `intersection` of the `ground bounding boxes` * the `intersection` of the `height` differences, divided by the union of the bounding boxes.

### Submission File

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

## Timeline

-   November 5, 2019 - Entry deadline. You must accept the competition rules before this date in order to compete.

-   November 5, 2019 - Team Merger deadline. This is the last day participants may join or merge teams.

-   November 12, 2019 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

Participants with the best score on the private leaderboard are eligible to receive:

-   1st Place - $12,000
-   2nd Place - $8,000
-   3rd Place - $5,000

Additional Awards and Opportunities\
One winner from the top three winning teams will be invited to present their results at our competition track at [NeurIPS 2019](https://neurips.cc/), including limited travel support. This winner will also be welcome to request a complimentary pass to NeurIPS 2019.

Top participants in this competition who are attending NeurIPS 2019 in Vancouver, Canada, will be invited to Lyft's competition party. Watch the forum for more details.

Lyft Level 5 is hiring! Members of top performing teams will be reviewed by our recruiting team for the chance to interview with Level 5. In the meantime, check out our [open roles](https://level5.lyft.com/#joinourteam).

## NeurIPS 2019

This Kaggle competition is part of Lyft Level 5's competition track that will be held at [NeurIPS 2019](https://neurips.cc/). Level 5 is returning for the second year to sponsor the conference, where you will have a chance to meet the team at their booth, or at their competition track. We will update the discussion forum with our NeurIPS booth and track location.

By attending the competition track, you'll get the chance to hear the winners of this Kaggle competition present their results. Attending the competition track is not required to participate in the Kaggle competition; however, winning participants will be asked to present their work.

NeurIPS is an annual academic conference in machine learning and computational science that attracts 8,000+ attendees. This year's conference will be held in Vancouver, Canada from December 8-14, 2019. We hope to see you there!

## Citation

Christy, Maggie, NikiNikatos, Phil Culliton, Vinay Shet, Vladimir Iglovikov. (2019). Lyft 3D Object Detection for Autonomous Vehicles. Kaggle. https://kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles


# Data

## Dataset Description

### What files do I need?

You will need the LIDAR, image, map and data files for both train and test (`test_images.zip`, `test_lidar.zip`, etc.). You may also need the `train.csv`, which includes the sample annotations in the form expected for submissions. The `sample_submission.csv` file contains all of the sample `Id`s for the test set.

The data files (`test_data.zip`, `train_data.zip`) are in JSON format.

### What should I expect the data format to be?

The data comes in the form of many interlocking tables and formats. The JSON files all contain single tables with identifying `tokens` that can be used to join with other files / tables. The images and lidar files all correspond to a sample in `sample_data.json`, and the `sample_token` from `sample_data.json` is the primary identifier used for the train and test samples.

The annotations in `train.csv` are in the following format:\
`center_x center_y center_z width length height yaw class_name`

-   `center_x`, `center_y` and `center_z` are the world coordinates of the center of the 3D bounding volume.
-   `width`, `length` and `height` are the dimensions of the volume.
-   `yaw` is the angle of the volume around the `z` axis (where `y` is forward/back, `x` is left/right, and `z` is up/down - making 'yaw' the direction the front of the vehicle / bounding box is pointing at while on the ground).
-   `class_name` is the type of object contained by the bounding volume.

### What am I predicting?

For the test samples, we're predicting the bounding volumes and classes of all of the objects in a given scene.

For example, in `sample_token` `97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c`, we might be predicting the presence of a car (with `confidence` `1.0`) and a bus (with `confidence` `0.5`). The prediction would look like this:

```
97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c,1.0 2742.152625996093 673.1631800662494 -18.6561112411676 1.834 4.609 1.648 2.619835541569646 car 0.5 2728.9634555684484 657.8296521874645 -18.54676216218047 1.799 4.348 1.728 -0.5425527100619654 bus
```

With all predicted objects on the same line.

Note that `confidence` values are inserted prior to `center_x center_y center_z width length height yaw class_name`.

### Files

-   **train_data.zip** and **test_data.zip** - contains JSON files with multiple tables. The most important is `sample_data.json`, which contains the primary identifiers used in the competition, as well as links to key image / lidar information.
-   **train_images.zip** and **test_images.zip** - contains .jpeg files corresponding to samples in `sample_data.json`
-   **train_lidar.zip** and **test_lidar.zip** - contains .jpeg files corresponding to samples in `sample_data.json`
-   **train_maps.zip** and **test_maps.zip** - contains maps of the entire sample area.
-   **train.csv** - contains all `sample_token`s in the train set, as well as annotations in the required format for all train set objects.
-   **sample_submission.csv** - contains all `sample_token`s in the test set, with empty predictions.