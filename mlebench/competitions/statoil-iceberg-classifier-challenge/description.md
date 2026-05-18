# Overview

## Description
![q7uAjTM](https://i.imgur.com/q7uAjTM.jpg)

Drifting icebergs present threats to navigation and activities in areas such as offshore of the East Coast of Canada.

Currently, many institutions and companies use aerial reconnaissance and shore-based support to monitor environmental conditions and assess risks from icebergs. However, in remote areas with particularly harsh weather, these methods are not feasible, and the only viable monitoring option is via satellite.

[Statoil](https://www.statoil.com/), an international energy company operating worldwide, has worked closely with companies like [C-CORE](https://www.c-core.ca/). C-CORE have been using satellite data for over 30 years and have built a computer vision based surveillance system. To keep operations safe and efficient, Statoil is interested in getting a fresh new perspective on how to use machine learning to more accurately detect and discriminate against threatening icebergs as early as possible.

In this competition, you’re challenged to build an algorithm that automatically identifies if a remotely sensed target is a ship or iceberg. Improvements made will help drive the costs down for maintaining safe working conditions.

## Evaluation

Submissions are evaluated on the [log loss](https://www.kaggle.com/wiki/LogLoss) between the predicted values and the ground truth.

## Submission File

For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1). The file should contain a header and have the following format:

```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
3aa99a38,0.9
etc.
```

## Prizes

See the Rules Section labeled "Prizes" for the terms on receiving prize money.

- 1st place - $25,000
- 2nd place - $15,000
- 3rd place - $10,000

## Timeline

- **January 16, 2018** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **January 16, 2018** - Team Merger deadline. This is the last day participants may join or merge teams.
- **January 23, 2018** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Background

![SO6VDSfm](https://storage.googleapis.com/kaggle-media/competitions/statoil/SO6VDSfm.jpg)

The remote sensing systems used to detect icebergs are housed on satellites over 600 kilometers above the Earth. The Sentinel-1 satellite constellation is used to monitor Land and Ocean. Orbiting 14 times a day, the satellite captures images of the Earth's surface at a given location, at a given instant in time. The C-Band radar operates at a frequency that "sees" through darkness, rain, cloud and even fog. Since it emits it's own energy source it can capture images day or night.

Satellite radar works in much the same way as blips on a ship or aircraft radar. It bounces a signal off an object and records the echo, then that data is translated into an image. An object will appear as a bright spot because it reflects more radar energy than its surroundings, but strong echoes can come from anything solid - land, islands, sea ice, as well as icebergs and ships. The energy reflected back to the radar is referred to as backscatter.

![NM5Eg0Q](https://storage.googleapis.com/kaggle-media/competitions/statoil/NM5Eg0Q.png)

![lhYaHT0](https://storage.googleapis.com/kaggle-media/competitions/statoil/lhYaHT0.png)

When the radar detects a object, it can't tell an iceberg from a ship or any other solid object. The object needs to be analyzed for certain characteristics - shape, size and brightness - to find that out. The area surrounding the object, in this case ocean, can also be analyzed or modeled. Many things affect the backscatter of the ocean or background area. High winds will generate a brighter background. Conversely, low winds will generate a darker background. The Sentinel-1 satellite is a side looking radar, which means it sees the image area at an angle (incidence angle). Generally, the ocean background will be darker at a higher incidence angle. You also need to consider the radar polarization, which is how the radar transmits and receives the energy. More advanced radars like Sentinel-1, can transmit and receive in the horizontal and vertical plane. Using this, you can get what is called a dual-polarization image.

For this contest you will see data with two channels: HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). This can play an important role in the object characteristics, since objects tend to reflect energy differently. Easy classification examples are see below. These objects can be visually classified. But in an image with hundreds of objects, this is very time consuming.

![8ZkRcp4](https://storage.googleapis.com/kaggle-media/competitions/statoil/8ZkRcp4.png)

![M8OP2F2](https://storage.googleapis.com/kaggle-media/competitions/statoil/M8OP2F2.png)

Here we see challenging objects to classify. We have given you the answer, but can you automate the answer to the question .... Is it a Ship or is it an Iceberg?

![AR4NDrK](https://storage.googleapis.com/kaggle-media/competitions/statoil/AR4NDrK.png)

![nXK6Vdl](https://storage.googleapis.com/kaggle-media/competitions/statoil/nXK6Vdl.png)

## Citation

Addison Howard, Brad Elliott, Carl Howell, chardy709@gmail.com, David Wade, Francesco S, Hallgeir, Harald W, Jan Richard Sagli, Kelley Dodge, Knut Sebastian, Mark McDonald, richh55, Wendy Kan. (2017). Statoil/C-CORE Iceberg Classifier Challenge. Kaggle. https://kaggle.com/competitions/statoil-iceberg-classifier-challenge
# Data
## Dataset Description

In this competition, you will predict whether an image contains a ship or an iceberg. The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

### Data fields

#### train.json, test.json

The data (`train.json`, `test.json`) is presented in `json` format.

The files consist of a list of images, and for each image, you can find the following fields:

- **id** - the id of the image
- **band_1, band_2** - the [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html) image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are **float** numbers with unit being [dB](https://en.wikipedia.org/wiki/Decibel). Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). More background on the satellite imagery can be found [here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background).
- **inc_angle** - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

#### sample_submission.csv
The submission file in the correct format:

- **id** - the id of the image
- **is_iceberg** - your predicted probability that this image is iceberg.