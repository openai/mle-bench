# Task

Given a recording, predict if it contains a right whale call (1) or noise (0).

# Metric

Area under the ROC curve.

# Submission Format

For every recording in the test set, submission files should contain two columns: clip and probability. The submission file should have a header.

```
clip,probability
20090404_000000_012s0ms_Test0.aif,0
20090404_000000_042s6ms_Test1.aif,0
20090404_000000_064s8ms_Test2.aif,0
20090404_000000_082s3ms_Test3.aif,0
20090404_000000_095s0ms_Test4.aif,0
```

# Dataset 

- **train2.zip** - all of the training data as .aif clips. If the file ends in "_1.aif" it was labeled a whale call, if it ends in "_0.aif", it was labeled noise.
- **test2.zip** -  all of the testing data as .aif clips. You should predict a probability for each of these files.