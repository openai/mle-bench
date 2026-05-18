# Task

Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

# Metric

Log loss.

# Submission Format

For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

# Dataset

The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.