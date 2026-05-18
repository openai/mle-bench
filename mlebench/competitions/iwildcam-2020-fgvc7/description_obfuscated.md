# Task

Classify species of images of animals from camera traps.

# Metric

Categorization accuracy.

# Submission Format

```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

# Dataset

Supplementary training data can be downloaded from https://github.com/visipedia/iwildcam_comp.

## Camera Trap Animal Detection Model

We are also providing a general animal detection model.

The model is a TensorFlow Faster-RCNN model with Inception-Resnet-v2 backbone and atrous convolution.

The model and sample code for running the detector over a folder of images can be found [here](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md).