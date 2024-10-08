# Overview

## Description

![Right whale](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3509/media/right_whale.jpg)*(right whale illustration courtesy of Pieter Folkens, ©2011)*

This competition complements the previously held [Marinexplore Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge), in which Cornell University provided data from a ship monitoring application termed "Auto Buoy", or AB Monitoring System. In the Marinexplore challenge we received solutions that exceeded 98% accuracy and will ultimately advance the process of automatically classifying North Atlantic Right Whales using the AB Monitoring Platform.

Since the results from the previous challenge proved so successful, we decided to extend the goals and consider applications that involve running algorithms on archival data recorded using portable hydrophone assemblies, otherwise referred to as Marine Autonomous Recording Unit (or MARU's). Since Cornell and its partners have been using the MARU for over a decade, a sizable collection of data has been accumulated. This data spans several ocean basins and covers a variety of marine mammal species.

Solutions to this challenge will be ported to a High Performance Computing (HPC) platform, being developed in part through funding provided by the Office of Naval Research (ONR grant N000141210585, Dugan, Clark, LeCun and Van Parijs). Together, Cornell will combine algorithms, HPC technologies and its data archives to explore data using highly accurate measuring tools. We encourage participants who developed prior solutions (through the collaboration with Marinexplore) to test them on this data.

The results will be presented at the [Workshop on Machine Learning for Bioacoustics](http://sabiod.univ-tln.fr/icml2013/challenge_description.html) at ICML 2013.

## Prizes

-   First place - $350
-   Second place - $150

## Evaluation

Submissions are judged on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic). 

In Matlab (using the stats toolbox):

```
[~, ~, ~, auc ] = perfcurve(true_labels, predictions, 1);
```

In R (using the verification package):

```
auc = roc.area(true_labels, predictions)
```

In python (using the metrics module of scikit-learn):

```
fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
auc = metrics.auc(fpr,tpr)
```

## Submission File

For every recording in the test set, submission files should contain two columns: clip and probability. The submission file should have a header.

```
clip,probability
20090404_000000_012s0ms_Test0.aif,0
20090404_000000_042s6ms_Test1.aif,0
20090404_000000_064s8ms_Test2.aif,0
20090404_000000_082s3ms_Test3.aif,0
20090404_000000_095s0ms_Test4.aif,0
```

### Citation

DeLMA, Will Cukierski. (2013). The ICML 2013 Whale Challenge - Right Whale Redux. Kaggle. https://kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux


# Data

## Dataset Description

You are given four days of training data and three days of testing data.  The task is to assign a probability that each recording in the test set contains a right whale call (1) or noise (0).

### File descriptions

- **train2.zip** - all of the training data as .aif clips. If the file ends in "_1.aif" it was labeled a whale call, if it ends in "_0.aif", it was labeled noise.
- **test2.zip** -  all of the testing data as .aif clips. You should predict a probability for each of these files.

To run within hardware constraints, solutions would ideally function with limited training history, meaning the algorithm would page in no more than five previous minutes of data at a time. Since we cannot enforce this requirement in this particular competition, we provide this info for those who want to optimize on "real world utility," as opposed to AUC.