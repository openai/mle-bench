# Overview

## Overview

### Description

As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.

Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.

In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

> **Recommended Tutorial**
>
> We highly recommend [Jesse Mostipak's Getting Started Tutorial](https://www.kaggle.com/jessemostipak/getting-started-tpus-cassava-leaf-disease) that walks you through making your very first submission step by step.

### Acknowledgements

The Makerere Artificial Intelligence (AI) Lab is an AI and Data Science research group based at Makerere University in Uganda. The lab specializes in the application of artificial intelligence and data science - including for example, methods from machine learning, computer vision and predictive analytics to problems in the developing world. Their mission is: "To advance Artificial Intelligence research to solve real-world challenges."

We thank the different experts and collaborators from National Crops Resources Research Institute (NaCRRI) for assisting in preparing this dataset.

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview/code-requirements) for details.

### Evaluation

#### Evaluation

Submissions will be evaluated based on their [categorization accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy).

#### Submission Format

The submission format for the competition is a csv file with the following format:

```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

### Timeline

- **February 11, 2021** - Entry deadline. You must accept the competition rules before this date in order to compete.

- **February 11, 2021** - Team Merger deadline. This is the last day participants may join or merge teams.

- **February 18, 2021** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

Leaderboard Prizes: Awarded based on private leaderboard ranking.

- 1st Place - \$8,000
- 2nd Place - \$4,000
- 3rd Place - \$3,000

TPU Star Prizes: Awarded to the most knowledgeable and helpful TPU experts in the community.

- Three \$1,000 Prizes

"TPU Star" prizes will be evaluated by experts from Google, on the following criteria:

1. Forum and notebook discussion/comment contributions
2. Quality of code samples in public notebooks and/or forums
3. Thoughtful analysis and explainability of code content

You may become eligible for these prizes either of two ways:

- Kaggle-identified: Kaggle will query all participants in the competition contributing upvoted public notebook(s) and making discussion contribution(s).
- Self-nominated: [Submit this form](https://www.kaggle.com/tpu-star-submission-form) by February 28, 2021 to self-nominate for consideration. At a minimum, you must have shared a public notebook which uses Kaggle's TPU integration on this competition's dataset.

> Be aware that because this is a code competition with a hidden test set, internet and TPUs cannot be enabled on your submission notebook. Therefore TPUs will only be available for training models. For a walk-through on how to train on TPUs and run inference/submit on GPUs, see our [TPU Docs](https://www.kaggle.com/docs/tpu#tpu6).

### Code Requirements

![Kerneler](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9` hours run-time
- GPU Notebook `<= 9` hours run-time
- [TPUs](https://www.kaggle.com/docs/tpu) will not be available for making submissions to this competition. You are still welcome to use them for training models. A walk-through for how to train on TPUs and run inference/submit on GPUs, see our [TPU Docs](https://www.kaggle.com/docs/tpu#tpu6).
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named "submission.csv"

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

### Citation

ErnestMwebaze, Jesse Mostipak, Joyce, Julia Elliott, Sohier Dane. (2020). Cassava Leaf Disease Classification. Kaggle. https://kaggle.com/competitions/cassava-leaf-disease-classification

# Data

## Dataset Description

Can you identify a problem with a cassava plant using a photo from a relatively inexpensive camera? This competition will challenge you to distinguish between several diseases that cause material harm to the food supply of many African countries. In some cases the main remedy is to burn the infected plants to prevent further spread, which can make a rapid automated turnaround quite useful to the farmers.

### Files

**[train/test]_images** the image files. The full set of test images will only be available to your notebook when it is submitted for scoring. Expect to see roughly 15,000 images in the test set.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.
