# Overview

## Overview

### Description

#### Goal of the Competition

Over 1.5 million spine fractures occur annually in the United States alone resulting in over 17,730 spinal cord injuries annually. The most common site of spine fracture is the cervical spine. There has been a rise in the incidence of spinal fractures in the elderly and in this population, fractures can be more difficult to detect on imaging due to superimposed degenerative disease and osteoporosis. Imaging diagnosis of adult spine fractures is now almost exclusively performed with computed tomography (CT) instead of radiographs (x-rays). Quickly detecting and determining the location of any vertebral fractures is essential to prevent neurologic deterioration and paralysis after trauma.

#### Context

RSNA has teamed with the [American Society of Neuroradiology (ASNR)](https://www.asnr.org/) and the [American Society of Spine Radiology (ASSR)](https://www.theassr.org/) to conduct an AI challenge competition exploring whether artificial intelligence can be used to aid in the detection and localization of cervical spine fractures.

To create the ground truth dataset, the challenge planning task force collected imaging data sourced from twelve sites on six continents, including approximately 3,000 CT studies. Spine radiology specialists from the ASNR and ASSR provided expert image level annotations these studies to indicate the presence, vertebral level and location of any cervical spine fractures.

In this challenge competition, you will try to develop machine learning models that match the radiologists' performance in detecting and localizing fractures to the seven vertebrae that comprise the cervical spine. Winners will be recognized at an event during the RSNA 2022 annual meeting.

For more information on the challenge, contact RSNA Informatics staff at <informatics@rsna.org>.

[A full set of acknowledgments can be found on this page](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/overview/acknowledgements).

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection/overview/code-requirements) for details.

### Evaluation

Submissions are evaluated using a weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

### Timeline

- **July 28, 2022** - Start Date.
- **October 20, 2022** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **October 20, 2022** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **October 27, 2022** - Final Submission Deadline.
- **November 8, 2022** - - Winners' Requirements Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video, method description.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

- 1st Place - \$6,000
- 2nd Place - \$5,000
- 3rd Place - \$4,000
- 4th - 8th Places - \$3,000 each

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA®) Annual Meeting, winners will be invited and strongly encouraged to attend the conference with waived fees, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the [competition rules](https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection/rules), in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, [solution packaging/delivery](https://www.kaggle.com/WinningModelDocumentationGuidelines), presentation to host), the host team also asks that you:

(i) create a short video presenting your approach and solution, and

(ii) publish a link to your open sourced code on the competition forum

(iii) (strongly suggested) make some version of your model publicly available for more hands-on testing purposes only. As an example of a hosted algorithm, please see <http://demos.md.ai/#/bone-age>.

### Code Requirements

#### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9` hours run-time
- GPU Notebook `<= 9` hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

### Acknowledgements

- Challenge Organizing Team
    - Adam Flanders MD
    - Errol Colak MD
    - Hui-Ming Lin HBSc
    - Ekim Gumeler MD
    - Jason Talbott MD PhD
    - Tyler Richards MD
    - Felipe C. Kitamura MD PhD
    - Luciano M Prevedello MD, MPH
    - Robyn Ball PhD
    - Amber Simpson PhD
    - Jayashree Kalpathy-Cramer PhD

- Data Contributors

Twelve research institutions from 9 countries and 6 continents provided large volumes of de-identified CT studies that were assembled to create the challenge dataset.

![data contributor map](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2022/image.png)

- Alfred Health, Australia
    - Meng Law
    - Adil Zia
    - Robin Lee

- Chiang Mai University, Thailand
    - Salita Angkurawaranon

- Hospital Regional Universitario de Málaga, Spain
    - Almudena Pérez
    - Isabel Gomez

- Kingston Health Sciences Centre, Canada
    - Johanna Ortiz Jimenez
    - Jacob Peoples
    - Mohammad Hamghalam
    - Amber Simpson

- Koç University, Türkiye
    - Hakan Dogan
    - Emre Altinmakas

- Prime Scan Alexandria, Egypt
    - Ayda Youssef
    - Yasser Mahfouz

- Stanford University, USA
    - Kristen Yeom
    - Yekaterina Shpanskaya
    - Allison Duh

- Thomas Jefferson University Hospital, USA
    - Baskaran Sundaram
    - Fam Ekladious
    - Albert Huang
    - Adam E Flanders

- Unity Health Toronto, Canada
    - Hui-Ming Lin
    - Errol Colak
    - Michael Brassil
    - Priscila Crivellaro

- University of Sarajevo, Bosnia and Herzegovina
    - Jasna Strika
    - Deniz Bulja
    - Nedim Kruscica

  Universidade Federal de São Paulo, Brazil
    - Felipe C. Kitamura
    - Nitamar Abdala
    - Eduardo Moreno Júdice de Mattos Farina

- University of Utah, USA
    - Tyler Richards
    - Michael Kushdilian

- Data Annotators

The challenge organizers wish to thank the American Society of Neuroradiology for recruiting its members to label the dataset used in the challenge. ASNR is the world's leading organization for the future of neuroradiology representing more than 5,300 radiologists, researchers, interventionalists, and imaging scientists.

![ANSR logo](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2022/ASNR%20logo.png) 
![ASSR logo](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2022/ASSR%20logo.jpg)

The American Society of Neuroradiology (ASNR) and the American Society of Spine Radiology (ASSR) organized a cadre of more than 40 volunteers to label exams for the challenge dataset.

- Gennaro D'Anna, MD ASST Ovest Milanese
- Allison M. Grayev, MD University of Wisconsin
- Fátima Hierro Hospital Pedro Hispano
- Michael D. Hollander, MD Norwalk Hospital
- Ichiro Ikuta, MD, MMSc Yale University School of Medicine
- Christie M. Lincoln, MD Baylor College of Medicine
- Lubdha M. Shah, MD University of Utah
- Achint K. Singh, MD UT Health San Antonio
- Nathan S. Doyle, MD McGovern Medical School
- Luis G. Colon Flores, MD NYU Langone Long Island Hospital
- Vikas Agarwal, MD University of Pittsburgh Medical Center
- Scott R. Andersen, MD Kaiser Permanente
- Katie Bailey, MD University of South Florida/James Haley VA
- Gagandeep Choudhary, MD, MBBS Oregon Health & Science University
- Sammy Chu, MD University of British Columbia
- Charlotte Y. Chung, MD, PhD NYU Langone Health
- Andrea S. Costacurta, MD Felippe Mattoso Clinic
- Muhammad Danial, MD Temple University
- Irene Dixe de Oliveira Santo, MD Yale New Haven Hospital
- Venkata Dola, MBBS George Washington University
- K. Jim Hsieh, MD University of Texas Medical Branch
- Adham Khalil, MD Johns Hopkins
- Neil U. Lall, MD Emory University/Childrens Healthcare
- Laurent Letourneau-Guillon, MD, MSc CHUM-Montreal
- David Russell Malin, MD Hampton Roads Radiology Associates
- J. Ryan Mason, DO, MPH University of Pittsburgh Medical Center
- Mariana Sanchez Montaño, MD Rh radiólogos
- Fanny E. Moron, MD Baylor College of Medicine
- Jaya Nath, MD Northport VA Medical Center
- Xuan V Nguyen, MD, PhD The Ohio State University
- Jacob Ormsby, MD, MBA University of New Mexico
- Mark C. Oswood, MD, PhD University of Minnesota
- Ozkan Ozsarlak, MD AZ Monica Hospitals
- Samuel Rogers, MD University of Arizona
- Jeffrey Rudie, MD, PhD UCSD and Scripps Clinic
- Anousheh Sayah, MD MGUH
- Eric D. Schwartz, MD St. Elizabeth's Medical Center
- Loizos Siakallis, MD Queen Square Institute of Neurology
- Neil B. Horner, MD Atlantic Health System
- Rogerio Jadjiski de Leao, MD UNIFESP

![MD.ai logo](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2022/MD.ai%20logo.png)

Special thanks to [MD.ai](https://md.ai/) for providing tooling for the data annotation process.-

### Data Resource Paper

Please cite this data resource paper if you plan to use this dataset.

Lin HM, Colak E, Richards T, Kitamura FC, Prevedello LM, Talbott J, Ball RL, Gumeler E, Yeom KW, Hamghalam M, Simpson AL, Strika J, Bulja D, Angkurawaranon S, Pérez-Lara A, Gómez-Alonso MI, Ortiz Jiménez J, Peoples JJ, Law M, Dogan H, Altinmakas E, Youssef A, Mahfouz Y, Kalpathy-Cramer J, Flanders AE, The RSNA-ASSR-ASNR Annotators and the Dataset Curation Contributors. The RSNA cervical spine fracture CT dataset. Radiology: Artificial Intelligence. 2023 Aug 30;5(5):e230034.\
URL: <https://pubs.rsna.org/doi/10.1148/ryai.230034>

### Citation

Adam Flanders, Chris Carr, Errol Colak, FelipeKitamura, MD, PhD, Hui Ming Lin, JeffRudie, John Mongan, Katherine Andriole, Luciano Prevedello, Michelle Riopel, Robyn Ball, Sohier Dane. (2022). RSNA 2022 Cervical Spine Fracture Detection. Kaggle. https://kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection

# Data

## Dataset Description

The goal of this competition is to identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient. Quickly detecting and determining the location of any vertebral fractures is essential to prevent neurologic deterioration and paralysis after trauma.

This competition uses a hidden test. When your submitted notebook is scored the actual test data (including a full length sample submission) will be made available to your notebook.

### Files

**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.
