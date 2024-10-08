# Overview

## Description

Tangles of kudzu overwhelm trees in Georgia while cane toads threaten habitats in over a dozen countries worldwide. These are just two invasive species of many which can have damaging effects on the environment, the economy, and even human health. Despite widespread impact, efforts to track the location and spread of invasive species are so costly that they’re difficult to undertake at scale.

Currently, ecosystem and plant distribution monitoring depends on expert knowledge. Trained scientists visit designated areas and take note of the species inhabiting them. Using such a highly qualified workforce is expensive, time inefficient, and insufficient since humans cannot cover large areas when sampling.

![invasive-species-monitoring.png](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3333/media/invasive-species-monitoring.png)

Because scientists cannot sample a large quantity of areas, some machine learning algorithms are used in order to predict the presence or absence of invasive species in areas that have not been sampled. The accuracy of this approach is far from optimal, but still contributes to approaches to solving ecological problems.

In this playground competition, Kagglers are challenged to develop algorithms to more accurately identify whether images of forests and foliage contain invasive hydrangea or not. Techniques from computer vision alongside other current technologies like aerial imaging can make invasive species monitoring cheaper, faster, and more reliable.

### Acknowledgments

Data providers: [Christian Requena Mesa](https://www.linkedin.com/in/requenac), Thore Engel, [Amrita Menon](https://www.linkedin.com/in/amrita-menon-12b58947/), Emma Bradley.

## Evaluation

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

### Submission File

For each image in the test set, you must predict a probability for the target variable on whether the image contains invasive species or not. The file should contain a header and have the following format:

```
name,invasive
2,0.5
5,0
6,0.2
etc.
```

## Citation

Crequena, Thore Engel, Wendy Kan. (2017). Invasive Species Monitoring. Kaggle. https://kaggle.com/competitions/invasive-species-monitoring

# Data

## Dataset Description

The data set contains pictures taken in a Brazilian national forest. In some of the pictures there is *Hydrangea*, a beautiful invasive species original of Asia. Based on the training pictures and the labels provided, the participant should predict the presence of the invasive species in the testing set of pictures.

## File descriptions

- **train.7z** - the training set (contains 2295 images).
- **train_labels.csv** - the correct labels for the training set.
- **test.7z** - the testing set (contains 1531 images), ready to be labeled by your algorithm.
- **sample_submission.csv** - a sample submission file in the correct format.

## Data fields

- **name** - name of the sample picture file (numbers)
- **invasive** - probability of the picture containing an invasive species. A probability of 1 means the species is present.