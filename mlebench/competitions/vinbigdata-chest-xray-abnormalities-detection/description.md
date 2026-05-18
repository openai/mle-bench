# Overview

## Description

When you have a broken arm, radiologists help save the day—and the bone. These doctors diagnose and treat medical conditions using imaging techniques like CT and PET scans, MRIs, and, of course, X-rays. Yet, as it happens when working with such a wide variety of medical tools, radiologists face many daily challenges, perhaps the most difficult being the chest radiograph. The interpretation of chest X-rays can lead to medical misdiagnosis, even for the best practicing doctor. Computer-aided detection and diagnosis systems (CADe/CADx) would help reduce the pressure on doctors at metropolitan hospitals and improve diagnostic quality in rural areas.

Existing methods of interpreting chest X-ray images classify them into a list of findings. There is currently no specification of their locations on the image which sometimes leads to inexplicable results. A solution for localizing findings on chest X-ray images is needed for providing doctors with more meaningful diagnostic assistance.

Established in August 2018 and funded by the Vingroup JSC, the Vingroup Big Data Institute (VinBigData) aims to promote fundamental research and investigate novel and highly-applicable technologies. The Institute focuses on key fields of data science and artificial intelligence: computational biomedicine, natural language processing, computer vision, and medical image processing. The medical imaging team at VinBigData conducts research in collecting, processing, analyzing, and understanding medical data. They're working to build large-scale and high-precision medical imaging solutions based on the latest advancements in artificial intelligence to facilitate effective clinical workflows.

In this competition, you’ll automatically localize and classify 14 types of thoracic abnormalities from chest radiographs. You'll work with a dataset consisting of 18,000 scans that have been annotated by experienced radiologists. You can train your model with 15,000 independently-labeled images and will be evaluated on a test set of 3,000 images. These annotations were collected via VinBigData's web-based platform, [VinLab](https://vindr.ai/vinlab). Details on building the dataset can be found in our recent paper [“VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations”](https://arxiv.org/pdf/2012.15029.pdf).

If successful, you'll help build what could be a valuable second opinion for radiologists. An automated system that could accurately identify and localize findings on chest radiographs would relieve the stress of busy doctors while also providing patients with a more accurate diagnosis.

### Acknowledgments

**Challenge Organizing Team**

- Ha Q. Nguyen, PhD - Vingroup Big Data Institute
- Hieu H. Pham, PhD - Vingroup Big Data Institute
- Nhan T. Nguyen, MSc - Vingroup Big Data Institute
- Dung B. Nguyen, BSc - Vingroup Big Data Institute
- Minh Dao, PhD - Vingroup Big Data Institute
- Van Vu, PhD - Vingroup Big Data Institute
- Khanh Lam, MD, PhD - Hospital 108
- Linh T. Le, MD, PhD - Hanoi Medical University Hospital

**Data Contributors**

The dataset used in this competition was created by assembling de-identified Chest X-ray studies provided by two hospitals in Vietnam: the Hospital 108 and the Hanoi Medical University Hospital.

## Evaluation

The challenge uses the standard PASCAL VOC 2010 [mean Average Precision (mAP)](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf) at IoU > 0.4.

### Submission File

Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID, `confidence` score, and bounding box in format `xmin ymin xmax ymax`. If you predict that there are NO objects in a given image, you should predict `14 1.0 0 0 1 1`, where `14` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

The submission file should contain a header and have the following format:

```
ID,TARGET
004f33259ee4aef671c2b95d54e4be68,14 1 0 0 1 1
004f33259ee4aef671c2b95d54e4be69,11 0.5 100 100 200 200 13 0.7 10 10 20 20
etc.
```

## Timeline

- **December 30, 2020** - Start Date.
- **March 23, 2021** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **March 23, 2021** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **March 30, 2021** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

**Leaderboard Prizes**

- 1st Place - $20,000
- 2nd Place - $14,000
- 3rd Place - $8,000

**Participants are strictly prohibited against hand-labeling the test set**, including use of any hand-labeling of the test set to inform model training or selection. The host will be thoroughly reviewing winning teams' full source code. Team(s) found violating this rule will be disqualified and removed from the competition.

---

**Special Prize**: $8,000

- Awarded to the highest performing team (based on private leaderboard score) whose team members are all Vietnamese citizens.
- Team members need not currently be Vietnam residents, but must be current valid passport-holders to qualify.
- A team may only win either the Leaderboard Prize or the Special Prize, but not both. If a Leaderboard Prize winner is also a Vietnamese team, we will move down the leaderboard to identify the next Vietnamese team who would qualify for the Special Prize.
- We will open up a form towards the end of the competition for eligible teams to submit for consideration.

## Citation

DungNB, Ha Q. Nguyen, Julia Elliott, KeepLearning, NguyenThanhNhan, Phil Culliton. (2020). VinBigData Chest X-ray Abnormalities Detection. Kaggle. https://kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection

# Data

## Dataset Description

In this competition, we are classifying common thoracic lung diseases and localizing critical findings. This is an object detection and classification problem.

For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "14 1 0 0 1 1" (14 is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

The images are in DICOM format, which means they contain additional data that might be useful for visualizing and classifying.

### Dataset information

The dataset comprises 18,000 postero-anterior (PA) CXR scans in DICOM format, which were de-identified to protect patient privacy.

All images were labeled by a panel of experienced radiologists for the presence of 14 critical radiographic findings as listed below:

```
0 - Aortic enlargement
1 - Atelectasis
2 - Calcification
3 - Cardiomegaly
4 - Consolidation
5 - ILD
6 - Infiltration
7 - Lung Opacity
8 - Nodule/Mass
9 - Other lesion
10 - Pleural effusion
11 - Pleural thickening
12 - Pneumothorax
13 - Pulmonary fibrosis
```

The "No finding" observation (`14`) was intended to capture the absence of all findings above.

*Note that a key part of this competition is working with ground truth from multiple radiologists.*

### Files

- **train.csv** - the train set metadata, with one row for each object, including a class and a bounding box. Some images in both test and train have multiple objects.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns

- `image_id` - unique image identifier
- `class_name` - the name of the class of detected object (or "No finding")
- `class_id` - the ID of the class of detected object
- `rad_id` - the ID of the radiologist that made the observation
- `x_min` - minimum X coordinate of the object's bounding box
- `y_min` - minimum Y coordinate of the object's bounding box
- `x_max` - maximum X coordinate of the object's bounding box
- `y_max` - maximum Y coordinate of the object's bounding box