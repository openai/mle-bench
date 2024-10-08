# Overview

## Overview

### Description

#### Hotel Recognition to Combat Human Trafficking

Victims of human trafficking are often photographed in hotel rooms as in the below examples. Identifying these hotels is vital to these trafficking investigations but poses particular challenges due to low quality of images and uncommon camera angles.

![Example investigative images.](https://cs.slu.edu/~astylianou/images/example_victim_images.png)

Even without victims in the images, hotel identification in general is a challenging fine-grained visual recognition task with a huge number of classes and potentially high intraclass and low interclass variation. In order to support research into this challenging task and create image search tools for human trafficking investigators, we created the TraffickCam mobile application, which allows every day travelers to submit photos of their hotel room. Read more about [TraffickCam on TechCrunch.](https://techcrunch.com/2016/06/25/traffickcam/)

Example images from one hotel in the TraffickCam dataset are shown below:

![Example TraffickCam images.](https://cs.slu.edu/~astylianou/images/example_traffickcam_images.png)

In this contest, competitors are tasked with identifying the hotel seen in test images from the TraffickCam dataset, which are based on a large gallery of training images with known hotel IDs.

Our team currently supports an image search system used at the National Center for Missing and Exploited Children in human trafficking investigations. Novel and interesting approaches have the potential to be incorporated in this search system.

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/hotel-id-2021-fgvc8/overview/code-requirements) for details.

### Evaluation

Submissions are evaluated according to the Mean Average Precision @ 5 (MAP@5):

$$
\text { MAP@5 }=\frac{1}{U} \sum_{u=1}^U \sum_{k=1}^{\min (n, 5)} P(k) \times \text{rel}(k)
$$

where $U$ is the number of images, $P(k)$ is the precision at cutoff $k$, $n$ is the number of predictions per image, and $\text{rel}(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant correct label, zero otherwise.

Once a correct label has been scored for *an observation*, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation. For example, if the correct label is `A` for an observation, the following predictions all score an average precision of `1.0`.

```
A B C D E
A A A A A
A B A C A
```

#### Submission File

For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The list should be sorted such that the first ID is considered the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

### Timeline

- March 10, 2021 - Competition start date.
- May 19, 2021 - Entry deadline. You must accept the competition rules before this date in order to compete.
- May 19, 2021 - Team Merger deadline. This is the last day participants may join or merge teams.
- May 26, 2021 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9` hours run-time
- GPU Notebook `<= 9` hours run-time
- Internet access disabled
- Pre-trained models are allowed
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

### CVPR 2021

This competition is part of the Fine-Grained Visual Categorization [FGVC8](https://sites.google.com/view/fgvc8) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2021](http://cvpr2021.thecvf.com/). A panel will review the top submissions for the competition based on the description of the methods provided. From this, a subset may be invited to present their results at the workshop. Attending the workshop is not required to participate in the competition; however, only teams that are attending the workshop will be considered to present their work.

There is no cash prize for this competition. CVPR 2021 will take place virtually. PLEASE NOTE: CVPR frequently sells out early, and we cannot guarantee CVPR registration after the competition's end. If you are interested in attending, please plan ahead.

You can see a list of all of the FGVC8 competitions [here](https://sites.google.com/view/fgvc8/competitions?authuser=0).

### Citation

Abby Stylianou, anjrash, Sohier Dane. (2021). Hotel-ID to Combat Human Trafficking 2021 - FGVC8. Kaggle. https://kaggle.com/competitions/hotel-id-2021-fgvc8

# Data

## Dataset Description

Identifying the location of a hotel room is a challenging problem of great interest for combating human trafficking. This competition provides a rich dataset of photos of hotel room interiors, without any people present, for this purpose.

Many of the hotels are independent or part of very small chains, where shared decor isn't a concern. However, the shared standards for their interior decoration for the larger chains means that many hotels can look quite similar at first glance. Identifying the chain can narrow the range of possibilities, but only down to a set that is much harder to tell apart and is still scattered across a wide geographic area.\
The real value lies in getting the number of candidates to a small enough number that a human investigator could follow up on all of them.

### Files

**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.
