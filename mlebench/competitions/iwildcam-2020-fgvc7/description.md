# Overview

## Description

Camera Traps (or Wild Cams) enable the automatic collection of large quantities of image data. Biologists all over the world use camera traps to monitor biodiversity and population density of animal species. We have recently been making strides towards automatic species classification in camera trap images. However, as we try to expand the scope of these models we are faced with an interesting problem: how do we train models that perform well on new (unseen during training) camera trap locations? Can we leverage data from other modalities, such as citizen science data and remote sensing data?

In order to tackle this problem, we have prepared a challenge where the training data and test data are from different cameras spread across the globe. The set of species seen in each camera overlap, but are not identical. The challenge is to classify species in the test cameras correctly. To explore multimodal solutions, we allow competitors to train on the following data: (i) our camera trap training set (data provided by WCS), (ii) iNaturalist 2017-2019 data, and (iii) multispectral imagery (from [Landsat 8](https://www.usgs.gov/land-resources/nli/landsat/landsat-8)) for each of the camera trap locations. On the competition [GitHub page](https://github.com/visipedia/iwildcam_comp) we provide the multispectral data, a taxonomy file mapping our classes into the iNat taxonomy, a subset of iNat data mapped into our class set, and a camera trap detection model (the MegaDetector) along with the corresponding detections.

If you use this dataset in publication, please cite:

```
@article{beery2020iwildcam,
    title={The iWildCam 2020 Competition Dataset},
    author={Beery, Sara and Cole, Elijah and Gjoka, Arvi},
    journal={arXiv preprint arXiv:2004.10340},
    year={2020}
}
```

This is an FGVCx competition as part of the [FGVC7](https://sites.google.com/view/fgvc7/home) workshop at [CVPR 2020](http://cvpr2020.thecvf.com/), and is sponsored by [Microsoft AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth) and [Wildlife Insights](https://www.wildlifeinsights.org/). There is a GitHub page for the competition [here](https://github.com/visipedia/iwildcam_comp). Please open an issue if you have questions or problems with the dataset.

You can find the iWildCam 2018 Competition [here](https://github.com/visipedia/iwildcam_comp/blob/master/2018/readme.md), and the iWildCam 2019 Competition [here](https://github.com/visipedia/iwildcam_comp/blob/master/2019/readme.md).

*Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.*

## Evaluation

### Evaluation

Submissions will be evaluated based on their categorization accuracy

### Submission Format

The submission format for the competition is a csv file with the following format:

```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

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

Arvi, Christine Kaeser-Chen, Elijah Cole, Elizabeth Park, Phil Culliton, sbeery. (2020). iWildCam 2020 - FGVC7. Kaggle. https://kaggle.com/competitions/iwildcam-2020-fgvc7

# Data

## Dataset Description

### Data Overview

The WCS training set contains 217,959 images from 441 locations, and the WCS test set contains 62,894 images from 111 locations. These 552 locations are spread across the globe.

You may also choose to use supplemental training data from the [iNaturalist 2017](https://github.com/visipedia/inat_comp/blob/master/2017/README.md), [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md) and [iNaturalist 2019](https://github.com/visipedia/inat_comp) competition datasets. As a courtesy, we have curated all the images from these datasets containing classes that might be in the test set and mapped them into the iWildCam categories. Note: these curated images come only from the iNaturalist 2017 and iNaturalist 2018 datasets because there are no common classes between the iNaturalist 2019 dataset and the WCS dataset. However, participants are still free to use the iNaturalist 2019 data.

This year we are providing [Landsat-8](https://www.usgs.gov/land-resources/nli/landsat/landsat-8) multispectral imagery for each camera location as supplementary data. In particular, each site is associated with a series of patches collected between 2013 and 2019. The patches are extracted from a "Tier 1" Landsat product, which consists only of data that meets certain geometric and radiometric quality standards. Consequently, the number of patches per site varies from 39 to 406 (median: 147). Each patch is 200x200x9 pixels, covering an area of 6km^2 at a resolution of 30 meters / pixel across 9 spectral bands. Note that all patches for a given site are registered, but are not centered exactly at the camera location to protect the integrity of the site.

The data can be downloaded from the competition [GitHub page](https://github.com/visipedia/iwildcam_comp).

### Camera Trap Animal Detection Model

We are also providing a general animal detection model which competitors are free to use as they see fit.

The model is a TensorFlow Faster-RCNN model with Inception-Resnet-v2 backbone and atrous convolution.

The model and sample code for running the detector over a folder of images can be found [here](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md).

We have run the detector over the three datasets, and provide the top 100 boxes and associated confidences along with the metadata for WCS.

See the competition [GitHub page](https://github.com/visipedia/iwildcam_comp) for further details.