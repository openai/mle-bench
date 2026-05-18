# Overview

## Description

Camera Traps (or Wild Cams) enable the automatic collection of large quantities of image data. Biologists all over the world use camera traps to monitor biodiversity and population density of animal species. We have recently been making strides towards automating the species classification challenge in camera traps, but as we try to expand the scope of these models from specific regions where we have collected training data to nearby areas we are faced with an interesting probem: how do you classify a species in a new region that you may not have seen in previous training data?

In order to tackle this problem, we have prepared a challenge where the training data and test data are from different regions, namely The American Southwest and the American Northwest. The species seen in each region overlap, but are not identical, and the challenge is to classify the test species correctly. To this end, we will allow training on our American Southwest data (from [CaltechCameraTraps](https://beerys.github.io/CaltechCameraTraps/)), on [iNaturalist 2017/2018](https://github.com/visipedia/inat_comp) data, and on simulated data generated from [Microsoft AirSim](https://github.com/Microsoft/AirSim). We have provided a taxonomy file mapping our classes into the iNat taxonomy.

This is an FGVCx competition as part of the [FGVC6](https://sites.google.com/view/fgvc6/home) workshop at [CVPR 2019](http://cvpr2019.thecvf.com/), and is sponsored by [Microsoft AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth). There is a github page for the competition [here](https://github.com/visipedia/iwildcam_comp). Please open an issue if you have questions or problems with the dataset.

If you use this dataset in publication, please cite:

```
@article{beery2019iwildcam,
 title={The iWildCam 2019 Challenge Dataset},
 author={Beery, Sara and Morris, Dan and Perona, Pietro},
 journal={arXiv preprint arXiv:1907.07617},
 year={2019}
}
```

*Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.*

## Evaluation

### Evaluation

Submissions will be evaluated based on their [macro F1 score](https://en.wikipedia.org/wiki/F1_score) - i.e. F1 will be calculated for each class of animal (including "empty" if no animal is present), and the submission's final score will be the unweighted mean of all class F1 scores.

## Submission Format

The submission format for the competition is a csv file with the following format:

```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

## Timeline

- **June 1, 2019** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **June 1, 2019** - Team Merger deadline. This is the last day participants may join or merge teams.
- **June 7, 2019** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## CVPR 2019

This competition is part of the Fine-Grained Visual Categorization [FGVC6](https://sites.google.com/view/fgvc6/home) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2019](http://cvpr2019.thecvf.com/). Top submissions for the competition will be invited to attend and present their results at the workshop. Attending the workshop is not required to participate in the competition, however only teams that are attending the workshop will be considered to present their work. All teams attending the workshop will be responsible for their own conference fees and transportation and housing costs.

## Citation

Maggie, Phil Culliton, sbeery. (2019). iWildCam 2019 - FGVC6. Kaggle. https://kaggle.com/competitions/iwildcam-2019-fgvc6

# Data

## Dataset Description

### Data Overview

The training set contains 196,157 images from 138 different locations in Southern California. You may also choose to use supplemental training data from [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017#data), [iNaturalist 2018](https://github.com/visipedia/inat_comp#data), iNaturalist 2019, and images simulated with [Microsoft AirSim](https://github.com/Microsoft/AirSim). As a courtesy, we have curated all the images from iNaturalist 2017/2018 containing classes that might be in the test set and mapped them into the iWildCam categories. This data (which we call "iNat Idaho") can be downloaded from our git page [here](https://github.com/visipedia/iwildcam_comp).

The test set contains 153,730 images from 100 locations in Idaho.

The competition task is to label each image with one of the following label ids:

```
name, id
empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22
```

Some classes may not occur in train or test.

## Camera Trap Animal Detection Model

We are also providing a [general animal detection model](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.pb), using Faster-RCNN with Inception-Resnet-v2 backbone and atrous convolution, which competitors are free to use as they see fit.

Sample code for running the detector over a folder of images can be found [here](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py).

We have run the detector over the three datasets and provide the top 100 detected boxes for each image with their associated confidence [here](https://github.com/visipedia/iwildcam_comp).

### Acknowledgements

Data is primarily provided by Erin Boydston (USGS), Justin Brown (NPS), iNaturalist, and the Idaho Department of Fish and Game (IDFG).