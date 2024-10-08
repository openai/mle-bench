# Overview

## Description

Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

![seedlings](https://storage.googleapis.com/kaggle-media/competitions/seedlings-classify/seedlings.png)

We're hosting this dataset as a Kaggle competition in order to give it wider exposure, to give the community an opportunity to experiment with different image recognition techniques, as well to provide a place to cross-pollenate ideas.

### Acknowledgments

We extend our appreciation to the Aarhus University Department of Engineering Signal Processing Group for hosting the [original data](https://vision.eng.au.dk/plant-seedlings-dataset/).

### Citation

[A Public Image Database for Benchmark of Plant Seedling Classification Algorithms](https://arxiv.org/abs/1711.05458)

## Evaluation

Submissions are evaluated on **MeanFScore**,

which at Kaggle is actually a [micro-averaged F1-score](https://en.wikipedia.org/wiki/F1_score).

Given positive/negative rates for each class *k*, the resulting score is computed this way:

$$
\begin{gathered}
\text { Precision }_{\text {micro }}=\frac{\sum_{k \in C} T P_k}{\sum_{k \in C} T P_k+F P_k} \\
\text { Recall }_{\text {micro }}=\frac{\sum_{k \in C} T P_k}{\sum_{k \in C} T P_k+F N_k}
\end{gathered}
$$

F1-score is the harmonic mean of precision and recall
$$
\text { MeanFScore }=F 1_{\text {micro }}=\frac{2 \text { Precision }_{\text {micro }} \text { Recall }_{\text {micro }}}{\text { Precision }_{\text {micro }}+\text { Recall }_{\text {micro }}}
$$

### Submission File

For each`file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Citation

inversion. (2017). Plant Seedlings Classification. Kaggle. https://kaggle.com/competitions/plant-seedlings-classification

# Data

## Dataset Description

You are provided with a training set and a test set of images of plant seedlings at various stages of grown. Each image has a filename that is its unique id. The dataset comprises 12 plant species. The goal of the competition is to create a classifier capable of determining a plant's species from a photo. The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

## File descriptions

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format