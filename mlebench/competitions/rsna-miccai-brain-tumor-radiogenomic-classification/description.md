# RSNA-MICCAI Brain Tumor Radiogenomic Classification

## Description

A malignant tumor in the brain is a life-threatening condition. Known as glioblastoma, it's both the most common form of brain cancer in adults and the one with the worst prognosis, with median survival being less than a year. The presence of a specific genetic sequence in the tumor known as MGMT promoter methylation has been shown to be a favorable prognostic factor and a strong predictor of responsiveness to chemotherapy.

![](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/image2.png)

Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of the cancer through imaging (i.e., radiogenomics) alone could be developed, this would potentially minimize the number of surgeries and refine the type of therapy required.

The Radiological Society of North America (RSNA) has teamed up with the Medical Image Computing and Computer Assisted Intervention Society (the MICCAI Society) to improve diagnosis and treatment planning for patients with glioblastoma. In this competition you will predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to train and test your model to detect for the presence of MGMT promoter methylation.

If successful, you'll help brain cancer patients receive less invasive diagnoses and treatments. The introduction of new and customized treatment strategies before surgery has the potential to improve the management, survival, and prospects of patients with brain cancer.

### Acknowledgments

**The Radiological Society of North America (RSNA®)** is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research and technological innovation.

RSNA provides high-quality educational resources, publishes five top peer-reviewed journals, hosts the world’s largest radiology conference and is dedicated to building the future of the profession through the RSNA Research & Education (R&E) Foundation, which has funded $66 million in grants since its inception. RSNA also supports and facilitates artificial intelligence (AI) research in medical imaging by sponsoring an ongoing series of AI challenge competitions.

**The Medical Image Computing and Computer Assisted Intervention Society (the MICCAI Society)** is dedicated to the promotion, preservation and facilitation of research, education and practice in the field of medical image computing and computer assisted medical interventions including biomedical imaging and medical robotics. The Society achieves this aim through the organization and operation of annual high quality international conferences, workshops, tutorials and publications that promote and foster the exchange and dissemination of advanced knowledge, expertise and experience in the field produced by leading institutions and outstanding scientists, physicians and educators around the world.

[A full set of acknowledgments can be found on this page](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/overview/acknowledgments).

![](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/sponsors.png)

## Evaluation

Submissions are evaluated on the [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

### Submission File

For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Timeline

- **July 13, 2021** - Start Date.
- **October 8, 2021** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **October 8, 2021** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **October 15, 2021** - Final Submission Deadline.
- **October 25, 2021** - Winners’ Requirements Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video, method description.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - $6,000
- 2nd Place - $5,000
- 3rd Place - $4,000
- 4th - 8th Places - $3,000 each

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA®) Annual Meeting, winners will be invited and strongly encouraged to attend the conference with waived fees, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the [competition rules](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/rules), in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, [solution packaging/delivery](https://www.kaggle.com/WinningModelDocumentationGuidelines), presentation to host), the host team also asks that you:

(i) create a short video presenting your approach and solution, and

(ii) publish a link to your open sourced code on the competition forum

## Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Acknowledgments

The dataset for this challenge has been collected from institutions around the world as part of a decade-long project to advance the use of AI in brain tumor diagnosis and treatment, the **Brain Tumor Segmentation (BraTS) challenge**. Running in parallel with this challenge, a challenge addressing segmentation represents the culmination of this effort.

- A comprehensive description of the both tasks of the RSNA-MICCAI Brain Tumor challenge can be found at: https://www.med.upenn.edu/cbica/brats2021/
- Participants interested in the segmentation task of this competition can find more info at: https://www.synapse.org/brats2021

### Challenge Organizing Team

*(in alphabetical order)*

- Spyridon Bakas, PhD - University of Pennsylvania, Philadelphia, PA, USA
- Ujjwal Baid, PhD - University of Pennsylvania, Philadelphia, PA, USA
- Evan Calabrese, MD, PhD - University of California San Francisco, CA, USA
- Christopher Carr - Radiological Society of North America (RSNA), Oak Brook, IL, USA
- Errol Colak, MD - Unity Health Toronto, Canada
- Keyvan Farahani, PhD - National Cancer Institute (NCI), National Institutes of Health (NIH), Bethesda, MD, USA
- Adam E. Flanders, MD - Thomas Jefferson University Hospital, Philadelphia, PA, USA
- Jayashree Kalpathy-Cramer, PhD - Massachusetts General Hospital, Boston, MA, USA
- Felipe C Kitamura, MD, MSc, PhD - Diagnósticos da América SA (Dasa) and Universidade Federal de São Paulo, Brazil
- Bjoern Menze, PhD - University of Zurich, Switzerland
- John Mongan, MD, PhD - University of California - San Francisco, CA, USA
- Luciano Prevedello, MD, MPH - The Ohio State University, Columbus, OH, USA
- Jeffrey Rudie, MD, PhD - University of California - San Francisco, CA, USA
- Russell Taki Shinohara, PhD - University of Pennsylvania, Philadelphia, PA, USA

### Data Contributors

*(in order of decreasing data contributions)*

- Christos Davatzikos, PhD, & Spyridon Bakas, PhD, & Chiharu Sako, PhD - University of Pennsylvania, Philadelphia, PA, USA
- John Mongan, MD, PhD, & Evan Calabrese, MD, PhD, & Jeff Rudie, MD, PhD, & Christopher Hess, MD, PhD, & Soonmee Cha, MD, & Javier Villanueva-Meyer, MD - University of California San Francisco, CA, USA
- John B. Freymann & Justin S. Kirby - on behalf of The Cancer Imaging Archive (TCIA), Cancer Imaging Program, NCI, NIH, USA
- Benedikt Wiestler, MD, & Bjoern Menze, PhD - Technical University of Munich, Germany
- Bjoern Menze, PhD - University of Zurich, Switzerland
- Errol Colak, MD & Priscila Crivellaro, MD - University of Toronto, Toronto, ON, Canada
- Rivka R. Colen, MD, & Aikaterini Kotrotsou, PhD - MD Anderson Cancer Center, TX, USA
- Daniel Marcus, PhD, & Mikhail Milchenko, PhD, & Arash Nazeri, MD - Washington University School of Medicine in St. Louis, MO, USA
- Hassan Fathallah-Shaykh, MD, PhD - University of Alabama at Birmingham, AL, USA
- Roland Wiest, MD - University of Bern, Switzerland
- Andras Jakab, MD, PhD - University of Debrecen, Hungary
- Marc-Andre Weber, MD - Heidelberg University, Germany
- Abhishek Mahajan, MD, & Ujjwal Baid, PhD - Tata Memorial Centre, Mumbai, India, & SGGS Institute of Engineering and Technology, Nanded, India

### Sponsors

Prize money for challenge winners is provided by:

- Intel Corporation
- Radiological Society of North American (RSNA)
- NeoSoma, Inc.

## Citation

Adam Flanders, Chris Carr, Evan Calabrese, FelipeKitamura, MD, PhD, inversion, JeffRudie, John Mongan, Julia Elliott, Luciano Prevedello, Michelle Riopel, sprint, Spyridon Bakas, Ujjwal. (2021). RSNA-MICCAI Brain Tumor Radiogenomic Classification. Kaggle. https://kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification

# Data

## Dataset Description

The competition data is defined by three cohorts: Training, Validation (Public), and Testing (Private). The “Training” and the “Validation” cohorts are provided to the participants, whereas the “Testing” cohort is kept hidden at all times, during and after the competition.

These 3 cohorts are structured as follows: Each independent case has a dedicated folder identified by a five-digit number. Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format. The exact mpMRI scans included are:

- Fluid Attenuated Inversion Recovery (FLAIR)
- T1-weighted pre-contrast (T1w)
- T1-weighted post-contrast (T1Gd)
- T2-weighted (T2)

Exact folder structure:

```
Training/Validation/Testing
│
└─── 00000
│   │
│   └─── FLAIR
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ ...
│   │
│   └─── T1w
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ ...
│   │
│   └─── T1wCE
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ ...
│   │
│   └─── T2w
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ .....
│
└─── 00001
│   │ ...
│
│ ...
│
└─── 00002
│   │ ...

```

## Files

- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

## Citation

If you reference or use the dataset in any form, include the following citation:

> U.Baid, et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification”, arXiv:2107.02314, 2021.
>