# Overview

## Description

That file you downloaded may contain hidden messages that aren't part of its regular contents. The same technology employed for digital watermarking is also misused by crime rings. Law enforcement must now use steganalysis to detect these messages as part of their investigations. Machine learning is an important tool in the discovery of this secret data.

![](https://storage.googleapis.com/kaggle-media/competitions/UTT/Kaggle_ALASKA.png)

Current methods produce unreliable results, raising false alarms. One reason for inaccuracy is the many different devices and processing combinations. Yet, detection models are trained on a homogeneous dataset. To increase accuracy, researchers must put data hidden within digital images "into the wild" (hence the name ALASKA) to mimic real world conditions.

In the competition, you'll create an efficient and reliable method to detect secret data hidden within innocuous-seeming digital images. Rather than limiting the data source, these images have been acquired with as many as 50 different cameras (from smartphone to full-format high end) and processed in different fashions. Successful entries will include robust detection algorithms with minimal false positives.

The IEEE WIFS (Workshop on Information Forensics and Security) is eager to make this happen again, as a follow up to the [ALASKA#1 Challenge](https://alaska.utt.fr/#HallOfFame). WIFS is an annual event where researchers gather to discuss emerging challenges, exchange fresh ideas, and share state-of-the-art results and technical expertise in the areas of information security and forensics. WIFS has teamed up with Troyes University of Technology, CRIStAL Lab, Lille University, and CNRS to enable more accurate steganalysis.

Law enforcement officers need better methods to combat criminals using hidden messages. The data science community and other researchers can help with better automated detection. More accurate methods could help catch criminals whose communications are hidden in plain sight.

The challenge is organized by Rémi COGRANNE (UTT), Patrick BAS (CRIStAL / CNRS) and Quentin Giboulot (UTT) ; in addition to Kaggle, we have been greatly helped by the following sponsors:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1951250%2F3a24a302d6a43d42087769d57048b566%2Flogo_UTT_CRIStAL.png?generation=1588208625932740&alt=media) ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1951250%2Fa5c6f8c179e51a4048b581191166425d%2FLogoCNRS_SPS.png?generation=1588208648755636&alt=media)

## Evaluation

In order to focus on reliable detection with an emphasis on low false-alarm rate, submissions are evaluated on the weighted AUC. To calculate the weighted AUC, each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

The figure below illustrates the different areas. This example also shows that the submission #2 has the lowest AUC but is ranked first since it outperforms its competitors for low False Positive Rate (FPR). On the opposite, the submission #3 has the highest AUC but would have been ranked third.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1951250%2Ff250ff6a4e04bac332fa14d539ed813e%2FKaggle.png?generation=1588207999884987&alt=media)

## Submission File

For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```

## Timeline

- July 13, 2020 - Entry deadline. You must accept the competition rules before this date in order to compete.

- July 13, 2020 - Team Merger deadline. This is the last day participants may join or merge teams.

- July 20, 2020 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - $12,000
- 2nd Place - $8,000
- 3rd Place - $5,000

## Citation

Addison Howard, GIBOULOT Quentin, PatrickFrenchie, Rémi Cogranne, Will Cukierski. (2020). ALASKA2 Image Steganalysis. Kaggle. https://kaggle.com/competitions/alaska2-image-steganalysis


# Data

## Dataset Description

This dataset contains a large number of unaltered images, called the "Cover" image, as well as corresponding examples in which information has been hidden using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).\
The goal of the competition is to determine which of the images in the test set (`Test/`) have hidden messages embedded.

Note that in order to make the competition more realistic the length of hidden messages (the payload) will not be provided. The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

## Files

- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.