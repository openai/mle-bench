# Overview

## Overview

### Description

#### Goal of the Competition

The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

Your work improving the automation of detection in screening mammography may enable radiologists to be more accurate and efficient, improving the quality and safety of patient care. It could also help reduce costs and unnecessary medical procedures.

#### Context

According to the WHO, breast cancer is the most commonly occurring cancer worldwide. In 2020 alone, there were 2.3 million new breast cancer diagnoses and 685,000 deaths. Yet breast cancer mortality in high-income countries has dropped by 40% since the 1980s when health authorities implemented regular mammography screening in age groups considered at risk. Early detection and treatment are critical to reducing cancer fatalities, and your machine learning skills could help streamline the process radiologists use to evaluate screening mammograms.

Currently, early detection of breast cancer requires the expertise of highly-trained human observers, making screening mammography programs expensive to conduct. A looming shortage of radiologists in several countries will likely worsen this problem. Mammography screening also leads to a high incidence of false positive results. This can result in unnecessary anxiety, inconvenient follow-up care, extra imaging tests, and sometimes a need for tissue sampling (often a needle biopsy).

The competition host, the Radiological Society of North America (RSNA) is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research, and technological innovation.

Your efforts in this competition could help extend the benefits of early detection to a broader population. Greater access could further reduce breast cancer mortality worldwide.

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/rsna-breast-cancer-detection/overview/code-requirements) for details.

### Evaluation

Submissions are evaluated using the [probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. You can find [a Python implementation here](https://www.kaggle.com/code/sohier/probabilistic-f-score).

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

#### Submission Format

For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

prediction_id,cancer\
0-L,0\
0-R,0.5\
1-L,1\
...

### Timeline

- **November 28, 2022** - Start Date.
- **February 20, 2023** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **February 20, 2023** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **February 27, 2023** - Final Submission Deadline.
- **April 14, 2023** - Winners' Requirements Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video, method description.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

- 1st Place - \$ 10,000
- 2nd Place - \$ 8,000
- 3rd Place - \$ 7,000
- 4th - 8th Place(s) - \$ 5,000

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA®) Annual Meeting, winners will be invited and strongly encouraged to attend the conference with waived fees, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the [competition rules](https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection/rules), in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, [solution packaging/delivery](https://www.kaggle.com/WinningModelDocumentationGuidelines), presentation to host), the host team also asks that you:

(i) create a short video presenting your approach and solution, and

(ii) publish a link to your open sourced code on the competition forum

(iii) (strongly suggested) make some version of your model publicly available for more hands-on testing purposes only. As an example of a hosted algorithm, please see <http://demos.md.ai/#/bone-age>.

### Code Requirements

![Kerneler-white-desc2_transparent.png](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9` hours run-time
- GPU Notebook `<= 9` hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

### Acknowledgements

#### Screening Mammography Breast Cancer Detection Challenge Acknowledgements

![RSNA Logo](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/RSNA%20Logo.svg)         

RSNA would like to thank the following individuals and organizations whose contributions made possible the 2023 RSNA Screening Mammography Breast Cancer Detection AI Challenge:

#### Challenge Organizing Team

- Katherine P. Andriole, PhD - MGH & BWH Center for Clinical Data Science
- Robyn Ball, PhD - The Jackson Laboratory
- Yan Chen, PhD - University of Nottingham
- Helen Frazer, MBBS - BreastScreen Victoria
- Tatiana Kelil, MD - University of California - San Francisco
- Felipe Kitamura, MD - Universidade Federal de São Paulo
- Jackson Kwok - BreastScreen Victoria
- Matthew Lungren, MD - Stanford University
- Ritse Mann, MD, PhD - Radboud University Medical Center
- John Mongan, MD, PhD - University of California - San Francisco
- Linda Moy, MD - New York University Grossman School of Medicine
- George Partridge - University of Nottingham
- Hari Trivedi, MD - Emory University
- Xin Wang, PhD - Radboud University Medical Center
- Luyan Yao, PhD - University of Nottingham
- Tianyu Zhang - Radboud University Medical Center

#### Data Contributors

Two research institutions provided large volumes of de-identified mammography studies and related clinical information that were assembled to create the challenge dataset.

![BreastScreen Victoria logo](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/BreastScreen%20Victoria%20logo_full-colour_Primary_Positive_CMYK%5B7%5D.jpg)         

*BreastScreen Victoria, Australia*

-   Helen Frazer, MBBS FRANZCR M.Epi
-   Michael S. Elliott, BEng (Hons)

![Hiti Lab logo](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/emory_hitti.png)         

*Emory University | Health Innovation and Translational Informatics (HITI) Lab - Hari Trivedi, MD*

-   Judy Gichoya, MD
-   Minjae Woo, PhD
-   Imon Banerjee, PhD
-   Jason Jeong, PhD
-   Beatrice Brown-Mulry, BS

![MD.ai logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2Fb54f8ef1c415f02a1376f7e880257814%2Fmdai-logo.png?generation=1568657775088597&alt=media)

Special thanks to *MD.ai* for providing tooling for the data annotation process.

![Kaggle logo](https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png)         

Special thanks to *Kaggle* for providing the competition platform, as well as design and technical support.

### Citation

Chris Carr, FelipeKitamura, MD, PhD, George Partridge, inversion, Jayashree Kalpathy-Cramer, John Mongan, Katherine Andriole, Lavender, Maryam Vazirabad, Michelle Riopel, Robyn Ball, Sohier Dane, Yan Chen. (2022). RSNA Screening Mammography Breast Cancer Detection. Kaggle. https://kaggle.com/competitions/rsna-breast-cancer-detection

# Data

## Dataset Description

**Note: The dataset for this challenge contains radiographic breast images of female subjects.**

The goal of this competition is to identify cases of breast cancer in mammograms from screening exams. It is important to identify cases of cancer for obvious reasons, but false positives also have downsides for patients. As millions of women get mammograms each year, a useful machine learning tool could help a great many people.

This competition uses a hidden test. When your submitted notebook is scored the actual test data (including a full length sample submission) will be made available to your notebook.

### Files

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission. Only the first few rows are available for download.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.
