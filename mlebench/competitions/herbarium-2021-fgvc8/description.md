# Overview

## Description

*The Herbarium 2021: Half-Earth Challenge* is to identify vascular plant specimens provided by the [New York Botanical Garden](https://www.nybg.org/) (NY), [Bishop Museum](https://www.bishopmuseum.org/) (BPBM), [Naturalis Biodiversity Center](https://www.naturalis.nl/en) (NL), [Queensland Herbarium](https://www.qld.gov.au/environment/plants-animals/plants/herbarium) (BRI), and [Auckland War Memorial Museum](https://www.aucklandmuseum.com/) (AK).

*The Herbarium 2021: Half-Earth Challenge* dataset includes more than **2.5M images** representing nearly **65,000 species** from the Americas and Oceania that have been aligned to a standardized plant list ([LCVP v1.0.2](https://www.nature.com/articles/s41597-020-00702-z)).

This dataset has a long tail; there are a minimum of 3 images per species. However, some species can be represented by more than 100 images. This dataset only includes vascular land plants which include lycophytes, ferns, gymnosperms, and flowering plants. The extinct forms of lycophytes are the major component of coal deposits, ferns are indicators of ecosystem health, gymnosperms provide major habitats for animals, and flowering plants provide almost all of our crops, vegetables, and fruits.

The teams with the most accurate models will be contacted with the intention of using them on the unnamed plant collections in the NYBG herbarium and then be assessed by the NYBG plant specialists for accuracy.

![Herbarium2021](https://i.postimg.cc/htpxH99f/Herbarium2021.png)

### Background

There are approximately 3,000 herbaria world-wide, and they are massive repositories of plant diversity data. These collections not only represent a vast amount of plant diversity, but since herbarium collections include specimens dating back hundreds of years, they provide snapshots of plant diversity through time. The integrity of the plant is maintained in herbaria as a pressed, dried specimen; a specimen collected nearly two hundred years ago by Darwin looks much the same as one collected a month ago by an NYBG botanist. All specimens not only maintain their morphological features but also include collection dates and locations, their reproductive state, and the name of the person who collected the specimen. This information, multiplied by millions of plant collections, provides the framework for understanding plant diversity on a massive scale and learning how it has changed over time. The models developed during this competition are an integral first step to speed the pace of species discovery and save the plants of the world.

There are approximately 400,000 known vascular plant species with an estimated 80,000 still to be discovered. Herbaria contain an overwhelming amount of unnamed and new specimens, and with the threats of climate change, we need new tools to quicken the pace of species discovery. This is more pressing today as a United Nations report indicates that more than one million species are at risk of extinction, and amid this dire prediction is a recent estimate that suggests plants are disappearing more quickly than animals. This year, we have expanded our curated herbarium dataset to vascular plant diversity in the Americas and Oceania.

The most accurate models will be used on the unidentified plant specimens in our herbarium and assessed by our taxonomists thereby producing a tool to quicken the pace of species discovery.

### About

This is an FGVC competition hosted as part of the [FGVC8](https://sites.google.com/view/fgvc8) workshop at [CVPR 2021](http://cvpr2021.thecvf.com/) and sponsored by [NYBG](https://www.nybg.org/).

Details of this competition are mirrored on the [github](https://github.com/visipedia/herbarium_comp) page. Please post in the forum or open an issue if you have any questions or problems with the dataset.

### Acknowledgements

The images are provided by the [New York Botanical Garden](https://www.nybg.org/), [Bishop Museum](https://www.bishopmuseum.org/), [Naturalis Biodiversity Center](https://www.naturalis.nl/en), [Queensland Herbarium](https://www.qld.gov.au/environment/plants-animals/plants/herbarium), and [Auckland War Memorial Museum](https://www.aucklandmuseum.com/).

![New York Botanical Garden, Bishop Museum, Naturalis Biodiversity Center, Queensland Herbarium, and Auckland War Memorial Museum logos](https://i.postimg.cc/fbSLBX54/Logos.png)

## Evaluation

Submissions are evaluated using the [macro F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

The F1 score is given by

$$
F_1=2 * \frac{\text { precision } * \text { recall }}{\text { precision }+ \text { recall }}
$$

where:

$$
\begin{gathered}
\text { precision }=\frac{T P}{T P+F P}, \\
\text { recall }=\frac{T P}{T P+F N}.
\end{gathered}
$$

In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged.

### Submission Format

For each image `Id`, you should predict the corresponding image label (`category_id`) in the `Predicted` column. The submission file should have the following format:

```
Id,Predicted
0,1
1,27
2,42
...
```

## Timeline

- **March 10, 2021** - Competition start date.
- **May 19, 2021** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **May 19, 2021** - Team Merger deadline. This is the last day participants may join or merge teams.
- **May 26, 2021** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## CVPR 2021

This competition is part of the Fine-Grained Visual Categorization [FGVC8](https://sites.google.com/view/fgvc8) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2021](http://cvpr2021.thecvf.com/). A panel will review the top submissions for the competition based on the description of the methods provided. From this, a subset may be invited to present their results at the workshop. Attending the workshop is not required to participate in the competition; however, only teams that are attending the workshop will be considered to present their work.

There is no cash prize for this competition. **CVPR 2021 will take place virtually.** PLEASE NOTE: CVPR frequently sells out early, we cannot guarantee CVPR registration after the competition's end. If you are interested in attending, please plan ahead.

You can see a list of all of the FGVC8 competitions [here](https://sites.google.com/view/fgvc8/competitions?authuser=0).

## Citation

Riccardo de Lutio, Titouan Lorieul, Walter Reade. (2021). Herbarium 2021 - Half-Earth Challenge - FGVC8. Kaggle. https://kaggle.com/competitions/herbarium-2021-fgvc8

# Data

## Dataset Description

### Data Overview

The training and test set contain images of herbarium specimens from nearly 65,000 species of vascular plants. Each image contains exactly one specimen. The text labels on the specimen images have been blurred to remove category information in the image.

The data has been approximately split 80%/20% for training/test. Each category has at least 1 instance in both the training and test datasets. Note that the test set distribution is slightly different from the training set distribution. The training set contains species with hundreds of examples, but the test set has the number of examples per species capped at a maximum of 10.

### Dataset Details

Each image has different image dimensions, with a maximum of 1000 pixels in the larger dimension. These have been resized from the original image resolution. All images are in JPEG format.

### Dataset Format

This dataset uses the [COCO dataset format](https://cocodataset.org/#format-data) with additional annotation fields. In addition to the species category labels, we also provide region and supercategory information.

The training set metadata (`train/metadata.json`) and test set metadata (`test/metadata.json`) are JSON files in the format below. Naturally, the test set metadata file omits the "annotations", "categories," and "regions" elements.

```
{
  "annotations" : [annotation],
  "categories" : [category],
  "images" : [image],
  "info" : info,
  "licenses": [license],
  "institutions": [region]
}

info {
  "year" : int,
  "version" : str,
  "url": str,
  "description" : str,
  "contributor" : str,
  "date_created" : datetime
}

image {
  "id" : int,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "license" : int
}

annotation {
  "id": int,
  "image_id": int,
  "category_id": int,
  # Region where this specimen was collected.
  "region_id": int
}

category {
  "id" : int,
  # Species name
  "name" : str,
  # We also provide the super-categories for each species.
  "family": str,
  "order": str
}

institution {
  "id": int
  "name": str
}

license {
  "id": 1,
  "name": str,
  "url": str
}

```

The training set images are organized in subfolders `train/images/<subfolder1>/<subfolder2>/<image id>.jpg`, where `<subfolder1>` combined with `<subfolder2>` corresponds to the `category_id`. For example, a training image with an `image_id` of `1104517` and a `category_id` of `00001`, can be found at `train/images/000/01/1104517.jpg`.

The test set images are organized in subfolders `test/images/<subfolder>/<image id>.jpg`, where `<subfolder>` corresponds to the integer division of the `image_id` by 1000. For example, a test image with and `image_id` of `8005`, can be found at `test/images/008/8005.jpg`.