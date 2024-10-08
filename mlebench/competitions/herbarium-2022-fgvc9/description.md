# Overview

## Description

[![My-Post.jpg](https://i.postimg.cc/15qZZfvt/My-Post.jpg)](https://postimg.cc/Xp4Pf7SS)

*The Herbarium 2022: Flora of North America* is a part of a project of the [New York Botanical Garden](https://www.nybg.org/) funded by the [National Science Foundation](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2054684&HistoricalAwards=false) to build tools to identify novel plant species around the world. The dataset strives to represent all known vascular plant taxa in North America, using images gathered from 60 different botanical institutions around the world.

In botany, a 'flora' is a complete account of the plants found in a geographic region. The dichotomous keys and detailed descriptions of diagnostic morphological features contained within a flora are used by botanists to determine which names to apply to plant specimens. This year's competition dataset aims to encapsulate the flora of North America so that we can test the capability of artificial intelligence to replicate this traditional tool ---a crucial first step to harnessing AI's potential botanical applications.

*The Herbarium 2022: Flora of North America* dataset comprises 1.05 M images of 15,501 vascular plants, which constitute more than 90% of the taxa documented in North America. Our dataset is constrained to include only vascular land plants (lycophytes, ferns, gymnosperms, and flowering plants).

Our dataset has a long-tail distribution. The number of images per taxon is as few as seven and as many as 100 images. Although more images are available, we capped the maximum number in an attempt to ensure sufficient but manageable training data size for competition participants.

## About

This is an FGVC competition hosted as part of the [FGVC9](https://sites.google.com/view/fgvc9) workshop at [CVPR 2022](http://cvpr2022.thecvf.com/) and sponsored by [NYBG](https://www.nybg.org/).

Details of this competition are mirrored on the [github](https://github.com/visipedia/herbarium_comp) page. Please post in the forum or open an issue if you have any questions or problems with the dataset.

## Acknowledgements

The images are provided by the [New York Botanical Garden](https://www.nybg.org/) and 59 other institutions around the world.\
![herb22banner](https://i.postimg.cc/g0DJMF52/output.png)

## Evaluation

Submissions are evaluated using the [macro F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). The F1 score is given by

$F_1=2 \frac{\text { precision } \cdot \text { recall }}{\text { precision }+ \text { recall }}$

where:

$\begin{gathered}\text { precision }=\frac{T P}{T P+F P}, \\ \text { recall }=\frac{T P}{T P+F N} .\end{gathered}$

In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged. #Submission Format For each image `Id`, you should predict the corresponding image label (`category_id`) in the `Predicted` column. The submission file should have the following format:
```
Id,Predicted
0,1
1,27
2,42
...
```

## Timeline

-   February 14, 2022 - Competition start date.
-   May 23, 2022 - Entry deadline. You must accept the competition rules before this date in order to compete.
-   May 23, 2022 - Team Merger deadline. This is the last day participants may join or merge teams.
-   May 30, 2022 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## CVPR 2022

This competition is part of the Fine-Grained Visual Categorization [FGVC9](https://sites.google.com/view/fgvc9) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2022](http://cvpr2022.thecvf.com/). A panel will review the top submissions for the competition based on the description of the methods provided. From this, a subset may be invited to present their results at the workshop. Attending the workshop is not required to participate in the competition; however, only teams that are attending the workshop will be considered to present their work.

There is no cash prize for this competition. PLEASE NOTE: CVPR frequently sells out early, we cannot guarantee CVPR registration after the competition's end. If you are interested in attending, please plan ahead.

You can see a list of all of the FGVC9 competitions [here](https://sites.google.com/view/fgvc9/competitions?authuser=0).

## Citation

Brendan Hogan, damon, inversion, John Park, Riccardo de Lutio. (2022). Herbarium 2022 - FGVC9. Kaggle. https://kaggle.com/competitions/herbarium-2022-fgvc9

# Dataset Description

## Data Overview

The training and test sets contain images of herbarium specimens from 15,501 species of vascular plants. Each image contains exactly one specimen. The text labels on the specimen images have been blurred to remove category information in the image.

The data has been approximately split 80%/20% for training/test. Each category has at least 1 instance in both the training and test datasets. Note that the test set distribution is slightly different from the training set distribution. The training set has a number of examples representing species capped at a maximum of 80.

## Dataset Details

### Images

Each image has different image dimensions, with a maximum of 1000 pixels in the larger dimension. These have been resized from the original image resolution. All images are in JPEG format.

### Hierarchical Structure of Classes `category_id`

In addition to the images, we also include a hierarchical taxonomic structure of `category_id`. The `categories` in the `training_metadata.json` contain three levels of hierarchical structure, `family` - `genus` - `species`, from the highest rank to the lowest rank. One can think about this as a directed graph, where families are the root nodes and the species are the leaf nodes. Please note that `species` is only unique under its parent node `genus`, which means that we can find multiple categories with the same `species` name under different `genus` names. This is due to the taxonomic nature that plants are named after, and the `genus`-`species` pair is always unique in our data.

### Phylogenetic Distances Among Genera

What makes this year's data unique is that we include a set of pairwise phylogenetic distances among genera, so that one could test if the difference in morphological features of plant taxa well correspond to their taxonomic distances. You can find the distance data with key name `distances` from `train_metadata.json`.

### Dataset Format

This dataset uses the [COCO dataset format](https://cocodataset.org/#format-data) with additional annotation fields. In addition to the species category labels, we also provide supercategory information.

The training set metadata (`train_metadata.json`) and test set metadata (`test_metadata.json`) are JSON files in the format below. Naturally, the test set metadata file omits the `annotations`, `categories`, and other elements.

```json
{
  "annotations" : [annotation],
  "categories" : [category],
  "genera" : [genus]
  "images" : [image],
  "distances" : [distance],
  "licenses" : [license],
  "institutions" : [institution]
}

annotation {
  "image_id" : int,
  "category_id" : int,
  "genus_id" : int,
  "institution_id" : int
}

image {
  "image_id" : int,
  "file_name" : str,
  "license" : int
}

category {
  "category_id" : int,
  "scientificName" : str,
  # We also provide a super-category for each species.
  "authors" : str, # correspond to 'authors' field in the wcvp
  "family" : str, # correspond to 'family' field in the wcvp
  "genus" : str, # correspond to 'genus' field in the wcvp
  "species" : str, # correspond to 'species' field in the wcvp
}

genera {
  "genus_id" : int,
  "genus" : str
}

distance {
  # We provide the pairwise evolutionary distance between categories (genus_id0 < genus_id1).
  "genus_id_x" : int,
  "genus_id_y" : int,
  "distance" : float
}

institution {
  "institution_id" : int
  "collectionCode" : str
}

license {
  "id" : int,
  "name" : str,
  "url" : str
}
```

The training set images are organized in subfolders `h22-train/images/<subfolder1>/<subfolder2>/<image_id>.jpg`, where `<subfolder1>` and `<subfolder2>` comes from the first three and the last two digits of the image_id. Image_id is a result of combination between `<category_id>` and unique numbers that differentiates images within plant taxa. Please be mindful that `category_id`s are unique, but not complete. Taxa are originally numbered from 1 to 15505, but the competition data has 15501 taxa because we lost four taxa during data cleaning process.

The test set images are organized in subfolders `test/images/<subfolder>/<image id>.jpg`, where `<subfolder>` corresponds to the integer division of the `image_id` by 1000. For example, a test image with and `image_id` of `8005`, can be found at `h22-test/images/008/test-008005.jpg`