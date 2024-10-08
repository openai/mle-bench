# Overview

## Overview

Start

Nov 16, 2020

Close

May 11, 2021

### Description

Our best estimates show there are over 7 billion people on the planet and 300 billion stars in the Milky Way galaxy. By comparison, the adult human body contains 37 *trillion* cells. To determine the function and relationship among these cells is a monumental undertaking. Many areas of human health would be impacted if we better understand cellular activity. A problem with this much data is a great match for the Kaggle community.

Just as the Human Genome Project mapped the entirety of human DNA, the [Human BioMolecular Atlas Program](https://hubmapconsortium.org/) (HuBMAP) is a major endeavor. Sponsored by the National Institutes of Health (NIH), HuBMAP is working to catalyze the development of a framework for mapping the human body at a level of glomeruli functional tissue units for the first time in history. Hoping to become one of the world's largest collaborative biological projects, HuBMAP aims to be an open map of the human body at the cellular level.

This competition, "Hacking the Kidney," starts by mapping the human kidney at single cell resolution.

Your challenge is to detect functional tissue units (FTUs) across different tissue preparation pipelines. An FTU is defined as a "three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block" ([de Bono, 2013](https://www.ncbi.nlm.nih.gov/pubmed/24103658)). The goal of this competition is the implementation of a successful and robust glomeruli FTU detector.

You will also have the opportunity to present your findings to a panel of judges for additional consideration. Successful submissions will construct the tools, resources, and cell atlases needed to determine how the relationships between cells can affect the health of an individual.

Advancements in HuBMAP will accelerate the world's understanding of the relationships between cell and tissue organization and function and human health. These datasets and insights can be used by researchers in cell and tissue anatomy, pharmaceutical companies to develop therapies, or even parents to show their children the magnitude of the human body.

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/hubmap-kidney-segmentation/overview/code-requirements) for details.

### Supervised ML Evaluation

This competition is evaluated on the mean [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 *|X \cap Y|}{|X|+|Y|},
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 1 when both $X$ and $Y$ are empty. The leaderboard score is the mean of the Dice coefficients for each image in the test set.

#### Submission File

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
img,pixels\
1,1 1 5 1\
2,1 1\
3,1 1\
etc.
```

### Judges Prize

Participants in this challenge will also have the opportunity to win prize money through the presentation of their submission. After entering the competition, participants can also elect to submit a presentation of their results for review by judges drawn from industry, academia, and government. Top selected submissions will have the opportunity to make a presentation to the judges panel to be evaluated on the below criteria.

Methodology (50 points) [Scientific Prize]

-   Are the statistical and modeling methods used to identify glomeruli in the PAS stained microscopy data appropriate for the task?
-   Are confidence scores and other metrics provided that help interpret the results achieved by the modeling methods?
-   Did the team validate their methods and algorithm implementations and provide information on algorithm performance and limitations?
-   Did the team provide any evidence that their method generalizes beyond this immediate task, for example to other FTUs such as alveoli in lungs or crypts in colon?
-   Did the team document their method and code appropriately?

Innovation and Applications (30 points) [Innovation Prize]

-   Did the team develop a creative or novel method to segment glomeruli?
-   Is the presented characterization of glomeruli useful for understanding individual differences, e.g., the impact of donor sex, age, or weight on the size, shape, or spatial distribution of glomeruli?
-   Did the team provide insights that would be useful for generating reference glomeruli for inclusion into a Human Reference Atlas?

Diversity and Presentation (30 points) [Diversity Prize]

-   Does the team embrace diversity and equity, welcoming team members of different ages, genders, ethnicities, and with multiple backgrounds and perspectives?
-   Did the authors effectively communicate the details of their method for segmenting glomeruli, and the quality and limitations of their results? For example, did they use data visualizations to present algorithm setup, run, results and/or to provide insight into the comparative performance of different methods? Were these visualizations effective at communicating insights about their approach and results?
-   Are the important results easily understood by the average person?

### Submission Instructions

You can make as many submissions as you like, but we will only consider your latest submission before the deadline of May 10.

All team members must be listed as collaborators on the submitted Notebook, and all team members must accept the competition rules before the submission deadline.

A valid submission will include:

*Notebook Analysis*: At least one notebook containing the analysis to support your proposal. All notebooks submitted must be made public on or before the submission deadline to be eligible. If submitting as a team, all team members must be listed as collaborators on all notebooks submitted.

Click [here](https://www.kaggle.com/hacking-the-kidney-judges-prize) to go to the Submission form

### Prizes

Accuracy Prize:

\$32,000 will be awarded to the three teams with the highest scores on the Kaggle Leaderboard at the conclusion of the competition:

-   First Place: \$18,000
-   Second Place: \$10,000
-   Third Place: \$4,000

Judges Prize:

\$28,000 will be awarded to teams that advance science and/or technology, are most innovative, and/or diverse as identified by the panel of judges through a presentation of their findings.

-   Scientific Prize: \$15,000
-   Innovation Prize: \$10,000
-   Diversity Prize: \$3,000

Teams can enter and win in multiple categories. Winning teams will receive cash prizes. If desired, teams can choose to have their winnings donated to a charity foundation.

### Timeline

*Updated March 9, 2021:*

-   May 3, 2021 - Entry deadline. You must accept the competition rules before this date in order to compete.

-   May 3, 2021 - Team Merger deadline. This is the last day participants may join or merge teams.

-   May 10, 2021 - Final submission deadline.

-   May 17, 2021 -- Selected teams present their solutions to Judges.

-   May 21, 2021  -- Judges announce winning teams at event with sponsors, participants, judges, organizers.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Organizers & Sponsors

The HuBMAP MC-IU team works in close collaboration with the HuBMAP HIVE teams and TMC-VU.

![Organizers](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Screen%20Shot%202020-11-16%20at%203.57.12%20PM.png)

![Judges](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Screen%20Shot%202020-11-05%20at%2012.08.58%20AM.png)

![Support](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Screen%20Shot%202020-11-05%20at%2012.09.04%20AM.png)

#### Innovation Sponsors

![Google](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Google.png)

![Roche](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Roche.png)

![Deloitte](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Deloitte.png)

![Deerfield](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/deerfield-logo-white.png)

#### Cyber Sponsors

![Pistoia](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Pistoia.png)

![CAS](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/CAS.png)

![Mavenware](https://storage.googleapis.com/kaggle-media/competitions/InnovDigi/Mavenware.png)

### Code Requirements

![Kerneler](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

#### This is a Code Competition

Submissions to this competition must be made through Notebooks. Please note that for this competition training is not required in Notebooks.

In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

-   CPU Notebook `<= 9` hours run-time
-   GPU Notebook `<= 9` hours run-time
-   [TPUs](https://www.kaggle.com/docs/tpu) will not be available for making submissions to this competition. You are still welcome to use them for training models. A walk-through for how to train on TPUs and run inference/submit on GPUs, see our [TPU Docs](https://www.kaggle.com/docs/tpu#tpu6).
-   No internet access enabled on submission
-   Freely & publicly available external data is allowed, including pre-trained models
-   Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ) for more information on how to submit.

### Citation

Addison Howard, Andy Lawrence, Bud Sims, Eddie Tinsley, Jarek Kazmierczak, Katy Borner, Leah Godwin, Marcos Novaes, Phil Culliton, Richard Holland, Rick Watson, Yingnan Ju. (2020). HuBMAP - Hacking the Kidney. Kaggle. https://kaggle.com/competitions/hubmap-kidney-segmentation

# Data

## Dataset Description

The HuBMAP data used in this hackathon includes 11 fresh frozen and 9 Formalin Fixed Paraffin Embedded (FFPE) PAS kidney images. Glomeruli FTU annotations exist for all 20 tissue samples; some of these will be shared for training, and others will be used to judge submissions.

There are over 600,000 glomeruli in each human kidney ([Nyengaard, 1992](https://onlinelibrary.wiley.com/doi/abs/10.1002/ar.1092320205)). Normal glomeruli typically range from 100-350μm in diameter with a roughly spherical shape ([Kannan, 2019](https://www.kireports.org/article/S2468-0249(19)30155-X/abstract)).

Teams are invited to develop segmentation algorithms that identify glomeruli in the PAS stained microscopy data. They are welcome to use other external data and/or pre-trained machine learning models in support of FTU segmentation. All data and all code used must be released under [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

### The Dataset

The dataset is comprised of very large (>500MB - 5GB) TIFF files. The training set has 8, and the public test set has 5. The private test set is larger than the public test set.

The training set includes annotations in both RLE-encoded and unencoded (JSON) forms. The annotations denote segmentations of glomeruli.

Both the training and public test sets also include anatomical structure segmentations. They are intended to help you identify the various parts of the tissue.

### File structure

The JSON files are structured as follows, with each feature having:

-   A `type` (`Feature`) and object type `id` (`PathAnnotationObject`). Note that these fields are the same between all files and do not offer signal.
-   A `geometry` containing a `Polygon` with `coordinates` for the feature's enclosing volume
-   Additional `properties`, including the name and color of the feature in the image.
-   The `IsLocked` field is the same across file types (locked for glomerulus, unlocked for anatomical structure) and is not signal-bearing.

Note that the objects themselves do NOT have unique IDs. The expected prediction for a given image is an RLE-encoded mask containing ALL objects in the image. The mask, as mentioned in the Evaluation page, should be binary when encoded - with `0` indicating the lack of a masked pixel, and `1` indicating a masked pixel.

`train.csv` contains the unique IDs for each image, as well as an RLE-encoded representation of the mask for the objects in the image. See the evaluation tab for details of the RLE encoding scheme.

`HuBMAP-20-dataset_information.csv` contains additional information (including anonymized patient data) about each image.
