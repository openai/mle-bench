# Overview

## Description

![Paddy Disease and Farmer](https://raw.githubusercontent.com/paddydoc/paddydoc.github.io/main/assets/img/paddy-disease-farmer.jpg)

### Problem Statement

Rice (Oryza sativa) is one of the staple foods worldwide. Paddy, the raw grain before removal of husk, is cultivated in tropical climates, mainly in Asian countries. Paddy cultivation requires consistent supervision because several diseases and pests might affect the paddy crops, leading to up to 70% yield loss. Expert supervision is usually necessary to mitigate these diseases and prevent crop loss. With the limited availability of crop protection experts, manual disease diagnosis is tedious and expensive. Thus, it is increasingly important to automate the disease identification process by leveraging computer vision-based techniques that achieved promising results in various domains.

### Objective

The main objective of this competition is to develop a machine or deep learning-based model to classify the given paddy leaf images accurately. We provide a training dataset of 10,407 (75%) labeled images across ten classes (nine disease categories and normal leaf). Moreover, we also provide additional metadata for each image, such as the paddy variety and age. Your task is to classify each paddy image in the given test dataset of 3,469 (25%) images into one of the nine disease categories or a normal leaf.

## Evaluation

### Evaluation

Submissions will be evaluated based on their [categorization accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy).

### Submission Format

The submission format for the competition is a csv file with the following format:

```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Timeline

### Competition Timeline

- March 31, 2022: Competition start date.
- July 24, 2022: Entry deadline. You must accept the competition rules before this date in order to compete.
- July 24, 2022: Team merger deadline. This is the last day participants may join or merge teams.
- August 31, 2022: Final submission deadline.

All deadlines are at 23:59 UTC on the corresponding day unless otherwise noted. We, as competition organizers, reserve the right to update the contest timeline if we deem it necessary.

## Acknowledgements

### Acknowledgements

We acknowledge the [Paddy Doctor](https://paddydoc.github.io/) project team at the Department of Computer Science and Engineering, Manonmaniam Sundaranar University, Tirunelveli, India, and its collaborators.

## Citation

Paddy Doctor, Pandarasamy Arjunan (Samy), Petchiammal. (2022). Paddy Doctor: Paddy Disease Classification. Kaggle. https://kaggle.com/competitions/paddy-disease-classification

# Data

## Dataset Description

We provide a training dataset of 10,407 (75%) labeled paddy leaf images across ten classes (nine diseases and normal leaf). We also provide additional metadata for each image, such as the paddy variety and age. Your task is to develop an accurate disease classification model using the training dataset and then classify each sample in the test dataset of 3,469 (25%) paddy leaf images into one of the nine diseases or normal leaf.

### Files

**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - This directory contains 10,407 training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - This directory contains 3,469 test set images.
