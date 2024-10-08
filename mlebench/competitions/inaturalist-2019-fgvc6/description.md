# Overview

## Description

![iNaturalist banner](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/0e1cef9d-568b-4cd0-9076-500fd0c282c4/image.png)

As part of the FGVC6 workshop at CVPR 2019 we are conducting the iNat Challenge 2019 large scale species classification competition, sponsored by Microsoft. It is estimated that the natural world contains several million species of plants and animals. Without expert knowledge, many of these species are extremely difficult to accurately classify due to their visual similarity. The goal  dof this competition is to push the state of the art in automatic image classification for real world data that features a large number of fine-grained categories.

Previous versions of the challenge have focused on classifying large numbers of species. This year features a smaller number of highly similar categories captured in a wide variety of situations, from all over the world. In total, the iNat Challenge 2019 dataset contains 1,010 species, with a combined training and validation set of 268,243 images that have been collected and verified by multiple users from iNaturalist.

Teams with top submissions, at the discretion of the workshop organizers, will be invited to present their work at the FGVC6 workshop. Participants who make a submission that beats the sample submission can fill out this [form](https://www.kaggle.com/GCP-Credits-CVPR2019) to receive $150 in Google Cloud credits.

![Microsoft Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_logo_%282012%29.svg/200px-Microsoft_logo_%282012%29.svg.png)

*Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.*

## Evaluation

We use top-1 classification error as the metric for this competition. For each image, an algorithm will produce 1 label. If the predicted label matches the ground truth label then the error for that image is 0, otherwise it is 1. The final score is the error averaged across all images.

### Submission File

For each image in the test set, you must predict 1 category label. However, we encourage you to predict more categories labels (sorted by confidence) so that we can analyze top-3 and top-5 performances. The csv file should contain a header and have the following format:

```
id,predicted
268243,71 108 339 341 560
268244,333 729 838 418 785
268245,690 347 891 655 755
```

The `id` column corresponds to the test image id. The `predicted` column corresponds to 1 category id. The first category id will be used to compute the metric. You should have one row for each test image.

## CVPR 2019

This competition is part of the Fine-Grained Visual Categorization [FGVC6](https://sites.google.com/view/fgvc6/home) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2019](http://cvpr2019.thecvf.com/). A panel will review the top submissions for the competition based on the description of the methods provided. From this, a subset may be selected to present their results at the workshop. Attending the workshop is not required to participate in the competition, however only teams that are attending the workshop will be considered to present their work.

There is no cash prize for this competition. Attendees presenting in person are responsible for all costs associated with travel, expenses, and fees to attend CVPR 2019.

## Timeline

- **June 3, 2019** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **June 3, 2019** - Team Merger deadline. This is the last day participants may join or merge teams.
- **June 10, 2019** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Citation

Grant Van Horn, macaodha, Maggie, Wendy Kan. (2019). iNaturalist 2019 at FGVC6. Kaggle. https://kaggle.com/competitions/inaturalist-2019-fgvc6

# Data

## Dataset Description

### File descriptions

- **train_val2019.tar.gz** - Contains the training and validation images in a directory structure following {iconic category name}/{category name}/{image id}.jpg .
- **train2019.json** - Contains the training annotations.
- **val2019.json** - Contains the validation annotations.
- **test2019.tar.gz** - Contains a single directory of test images.
- **test2019.json** - Contains test image information.
- **kaggle_sample_submission.csv** - A sample submission file in the correct format.

### Image Format

All images have been saved in the JPEG format and have been resized to have a maximum dimension of 800 pixels.

### Annotation Format

We follow the annotation format of the [COCO dataset][2] and add additional fields. The annotations are stored in the [JSON format][3] and are organized as follows:

```json
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation],
  "licenses" : [license]
}

info{
  "year" : int,
  "version" : str,
  "description" : str,
  "contributor" : str,
  "url" : str,
  "date_created" : datetime,
}

image{
  "id" : int,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "license" : int,
  "rights_holder" : str
}

category{
  "id" : int,
  "name" : str,
  "kingdom" : str,
  "phylum" : str,
  "class" : str,
  "order" : str,
  "family" : str,
  "genus" : str
}

annotation{
  "id" : int,
  "image_id" : int,
  "category_id" : int
}

license{
  "id" : int,
  "name" : str,
  "url" : str
}
```