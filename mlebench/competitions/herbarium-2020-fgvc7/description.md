# Overview

## Description

The Herbarium 2020 FGVC7 Challenge is to identify vascular plant species from a large, long-tailed collection herbarium specimens provided by the [New York Botanical Garden](https://www.nybg.org/plant-research-and-conservation/) (NYBG).

The Herbarium 2020 dataset contains over 1M images representing over 32,000 plant species. This is a dataset with a long tail; there are a minimum of 3 specimens per species. However, some species are represented by more than a hundred specimens. This dataset only contains vascular land plants which includes lycophytes, ferns, gymnosperms, and flowering plants. The extinct forms of lycophytes are the major component of coal deposits, ferns are indicators of ecosystem health, gymnosperms provide major habitats for animals, and flowering plants provide all of our crops, vegetables, and fruits.

![specimen1](https://raw.githubusercontent.com/visipedia/herbarium_comp/master/2020/assets/specimen1.jpg)
![specimen2](!https://raw.githubusercontent.com/visipedia/herbarium_comp/master/2020/assets/specimen2.jpg)
![specimen3](https://raw.githubusercontent.com/visipedia/herbarium_comp/master/2020/assets/specimen3.jpg)
![specimen4](https://raw.githubusercontent.com/visipedia/herbarium_comp/master/2020/assets/specimen4.jpg)
![specimen5](https://raw.githubusercontent.com/visipedia/herbarium_comp/master/2020/assets/specimen5.jpg)

The teams with the most accurate models will be contacted, with the intention of using them on the un-named plant collections in the NYBG herbarium collection, and assessed by the NYBG plant specialists.

### Background

The New York Botanical Garden (NYBG) herbarium contains more than 7.8 million plant and fungal specimens. Herbaria are a massive repository of plant diversity data. These collections not only represent a vast amount of plant diversity, but since herbarium collections include specimens dating back hundreds of years, they provide snapshots of plant diversity through time. The integrity of the plant is maintained in herbaria as a pressed, dried specimen; a specimen collected nearly two hundred years ago by Darwin looks much the same as one collected a month ago by an NYBG botanist. All specimens not only maintain their morphological features but also include collection dates and locations, and the name of the person who collected the specimen. This information, multiplied by millions of plant collections, provides the framework for understanding plant diversity on a massive scale and learning how it has changed over time.

### About

This is an FGVC competition hosted as part of the [FGVC7](https://sites.google.com/corp/view/fgvc7/home) workshop at [CVPR 2020](http://cvpr2020.thecvf.com/) and sponsored by [NYBG](https://www.nybg.org/plant-research-and-conservation/).

Details of this competition are mirrored on the [github page](https://github.com/visipedia/herbarium_comp). Please post in the forum or open an issue if you have any questions or problems with the dataset.

## Evaluation

Submissions are evaluated using the [macro F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

F1 is calculated as follows:

$$
F_1=2 * \frac{\text { precision } * \text { recall }}{\text { precision }+ \text { recall }}
$$

where:

$$
\begin{gathered}
\text { precision }=\frac{T P}{T P+F P} \\
\text { recall }=\frac{T P}{T P+F N}
\end{gathered}
$$

In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged.

### Submission Format

For each image `Id`, you should predict the corresponding image label ("category_id") in the `Predicted` column. The submission file should have the following format:

```
Id,Predicted
0,0
1,27
2,42
...
```

## Timeline

- **March 9, 2020** - Competition start date.
- **May 19, 2020** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **May 19, 2020** - Team Merger deadline. This is the last day participants may join or merge teams.
- **May 26, 2020** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## CVPR 2020

This competition is part of the Fine-Grained Visual Categorization [FGVC7](https://sites.google.com/view/fgvc7/home) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2020](http://cvpr2020.thecvf.com/). A panel will review the top submissions for the competition based on the description of the methods provided. From this, a subset may be invited to present their results at the workshop. Attending the workshop is not required to participate in the competition, however only teams that are attending the workshop will be considered to present their work.

There is no cash prize for this competition. Attendees presenting in person are responsible for all costs associated with travel, expenses, and fees to attend CVPR 2020. PLEASE NOTE: CVPR frequently sells out early, we cannot guarantee CVPR registration after the competition's end. If you are interested in attending, please plan ahead.

You can see a list of all of the FGVC7 competitions [here](https://sites.google.com/corp/view/fgvc7/competitions).

## Citation

Christine Kaeser-Chen, Kiat Chuan Tan, Walter Reade, Maggie Demkin. (2020). Herbarium 2020 - FGVC7. Kaggle. https://kaggle.com/competitions/herbarium-2020-fgvc7

# Data

## Dataset Description

### Data Overview

The training and test set contain images of herbarium specimens, from over 32,000 species of vascular plants. Each image contains exactly one specimen. The text and barcode labels on the specimen images have been blurred to remove category information in the image.

The data has been approximately split 80%/20% for training/test. Each category has at least 1 instance in both the training and test datasets. Note that the test set distribution is slightly different from the training set distribution. The training set contains species with hundreds of examples, but the test set has the number of examples per species capped at a maximum of 10.

### Dataset Details

Each image has different image dimensions, with a maximum of 1000 pixels in the larger dimension. These have been resized from the original image resolution. All images are in JPEG format.

### Dataset Format

This dataset uses the [COCO dataset format](http://cocodataset.org/#format-data) with additional annotation fields. In addition to the species category labels, we also provide region and supercategory information.

The training set metadata (`train/metadata.json`) and test set metadata (`test/metadata.json`) are JSON files in the format below. Naturally, the test set metadata file omits the "annotations", "categories" and "regions" elements.

```
{
  "annotations" : [annotation],
  "categories" : [category],
  "images" : [image],
  "info" : info,
  "licenses": [license],
  "regions": [region]
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
  "genus": str
}

region {
  "id": int
  "name": str
}

license {
  "id": 1,
  "name": str,
  "url": str
}
```

The training set images are organized in subfolders `train/<subfolder1>/<subfolder2>/<image id>.jpg`.

The test set images are organized in subfolders `test/<subfolder>/<image id>.jpg`.