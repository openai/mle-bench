# Overview

## Description

Five times more deadly than the flu, COVID-19 causes significant morbidity and mortality. Like other pneumonias, pulmonary infection with COVID-19 results in inflammation and fluid in the lungs. COVID-19 looks very similar to other viral and bacterial pneumonias on chest radiographs, which makes it difficult to diagnose. Your computer vision model to detect and localize COVID-19 would help doctors provide a quick and confident diagnosis. As a result, patients could get the right treatment before the most severe effects of the virus take hold.

![chest radiograph](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2Fd514aaf604bc9667b518b232a77d1aa7%2FCXR%20image1.jpg?generation=1620769201081719&alt=media)

Currently, COVID-19 can be diagnosed via polymerase chain reaction to detect genetic material from the virus or chest radiograph. However, it can take a few hours and sometimes days before the molecular test results are back. By contrast, chest radiographs can be obtained in minutes. While guidelines exist to help radiologists differentiate COVID-19 from other types of infection, their assessments vary. In addition, non-radiologists could be supported with better localization of the disease, such as with a visual bounding box.

As the leading healthcare organization in their field, the Society for Imaging Informatics in Medicine (SIIM)'s mission is to advance medical imaging informatics through education, research, and innovation. SIIM has partnered with the Foundation for the Promotion of Health and Biomedical Research of Valencia Region (FISABIO), Medical Imaging Databank of the Valencia Region (BIMCV) and the Radiological Society of North America (RSNA) for this competition.

In this competition, you'll identify and localize COVID-19 abnormalities on chest radiographs. In particular, you'll categorize the radiographs as negative for pneumonia or typical, indeterminate, or atypical for COVID-19. You and your model will work with imaging data and annotations from a group of radiologists.

If successful, you'll help radiologists diagnose the millions of COVID-19 patients more confidently and quickly. This will also enable doctors to see the extent of the disease and help them make decisions regarding treatment. Depending upon severity, affected patients may need hospitalization, admission into an intensive care unit, or supportive therapies like mechanical ventilation. As a result of better diagnosis, more patients will quickly receive the best care for their condition, which could mitigate the most severe effects of the virus.

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/siim-covid19-detection/overview/code-requirements) for details.

### Host Organizations

#### FISABIO, The Foundation for the Promotion of Health and Biomedical Research of Valencia Region

The Foundation for the Promotion of Health and Biomedical Research of Valencia Region, FISABIO, is a non-profit scientific and healthcare entity, whose primary purpose is to encourage, to promote and to develop scientific and technical health and biomedical research in Valencia Region. FISABIO integrates and manages the Health Research Map of the Centre for Public Health Research, Dr. Peset University Hospital Foundation, Alicante University General Hospital Foundation, Elche University General Hospital Foundation, and the Mediterranean Ophthalmological Foundation. The BIMCV facility is connected with a multi-level vendor neutral archive (VNA). The imaging population facility is storing data from the Valencia Region, which accounts for more than 5.1 million habitants.

#### Radiological Society of North America (RSNA)

The Radiological Society of North America (RSNA) is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research and technological innovation.

RSNA provides high-quality educational resources, publishes five top peer-reviewed journals, hosts the world's largest radiology conference and is dedicated to building the future of the profession through the RSNA Research & Education (R&E) Foundation, which has funded $66 million in grants since its inception. RSNA also supports and facilitates artificial intelligence (AI) research in medical imaging by sponsoring an ongoing series of AI challenge competitions.

## Evaluation

The challenge uses the standard PASCAL VOC 2010 [mean Average Precision (mAP) ](https://storage.googleapis.com/kaggle-media/competitions/SIIM2021/VOC2012_doc.pdf)at IoU > `0.5`. Note that the linked document describes VOC 2012, which differs in some minor ways (e.g. there is no concept of "difficult" classes in VOC 2010). The P/R curve and AP calculations remain the same.

In this competition, we are making predictions at both a study (multi-image) and image level.

#### Study-level labels

Studies in the test set may contain more than one label. They are as follows:

> "negative", "typical", "indeterminate", "atypical"

Please see the Data page for further details.

For each study in the test set, you should predict at least one of the above labels. The format for a given label's prediction would be a class ID from the above list, a `confidence` score, and `0 0 1 1` is a one-pixel bounding box.

#### Image-level labels

Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID of "opacity", a `confidence` score, and bounding box in format `xmin ymin xmax ymax`. If you predict that there are NO objects in a given image, you should predict `none 1.0 0 0 1 1`, where `none` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

### Submission File

The submission file should contain a header and have the following format:

```
Id,PredictionString
2b95d54e4be65_study,negative 1 0 0 1 1
2b95d54e4be66_study,typical 1 0 0 1 1
2b95d54e4be67_study,indeterminate 1 0 0 1 1 atypical 1 0 0 1 1
2b95d54e4be68_image,none 1 0 0 1 1
2b95d54e4be69_image,opacity 0.5 100 100 200 200 opacity 0.7 10 10 20 20
etc.
```

## Timeline

-   May 17, 2021 - Start Date.

-   August 2, 2021 - Entry Deadline. You must accept the competition rules before this date in order to compete.

-   August 2, 2021 - Team Merger Deadline. This is the last day participants may join or merge teams.

-   August 9, 2021 - Final Submission Deadline.

-   September 19-20, 2021 - Winners' Showcase at SIIM CMIMI 2021

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

**Leaderboard Prizes**: Awarded on the basis of private leaderboard rank. Only selected submissions will be ranked on the private leaderboard.

-   1st Place - $30,000
-   2nd Place - $20,000
-   3rd Place - $10,000
-   4th Place - $8,000
-   5th Place - $7,000
-   6th - 10th Places - $5,000 each

**Student Team Prize**: Awarded to the top-scoring eligible Student Team on the basis of private leaderboard rank. *To be fulfilled by HP's Fulfillment Agency*

One HP ZBook Studio G8 data science workstation will be awarded to each member of the winning Student Team. Each ZBook is valued at $6,000.

### Important Requirements

-   As this competition is aimed at the advancement of research using machine learning in medical imaging, the hosts require that winners fulfill the obligations stated in the [Competition Rules, Section A.3.](https://www.kaggle.com/c/siim-covid19-detection/rules) in order to retain their leaderboard position. If a team is unwilling to fulfill these obligations and declines their prize, they will also be disqualified from the competition and removed from the Competition leaderboard.

-   In addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, solution packaging/delivery), the host team also asks that you:\
    (i) provide a video presentation to be shared with the SIIM community,\
    (ii) publish a link to your open sourced code on the competition forum, and\
    (iii) *[optional]* create a publicly available demo version of the model for more hands-on testing purposes. As an example of a hosted algorithm, please see <http://demos.md.ai/#/bone-age>.

-   To be eligible to earn the Student Team Prize, in addition to meeting the eligibility rules specified in Section 2 (Eligibility) of the [Competition Rules](https://www.kaggle.com/c/siim-covid19-detection/rules), 50% or more of the members of a Student Team (i.e., 3 out of 5) must also be:\
    (i) currently enrolled as a full-time student (undergraduate or graduate) at a college, university, high school, secondary school or equivalent educational institution\
    (ii) able to provide proof of your current enrollment before you may receive a prize. If you cannot provide documentary proof that the required percentage of student members of a Student Team are currently enrolled as a full-time students, the Student Team may be disqualified or the team member without documentation may be disqualified, at the discretion of the Competition Sponsor.

-   We will open up a submission form to self-nominate your eligibility for the Student Team Prize at the end of the competition. It is possible to win both the Student Team Prize and a Leaderboard Prize. But to be considered for the Student Prize, you must self-nominate through the submission form.

-   All teams, regardless of place, are also strongly encouraged and invited to open source their code to the Model Zoo. [Contribute to the Model Zoo through this link](https://straiti3.org/blog/index.php/welcome-how-can-we-help/). Your contributions will greatly advance the diagnosis and treatment of COVID-19.

## Code Requirements

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

-   CPU Notebook <= 9 hours run-time
-   GPU Notebook <= 9 hours run-time
-   Internet access disabled
-   Freely & publicly available external data is allowed, including pre-trained models
-   Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Call For Models

### Support Research - Call for Open Source AI Models

Reproducible science requires reproducible review. AI is poised to transform the medical imaging process from diagnosis through intervention, but these technologies are not reaching patients due to complex, non-scalable, and non-verifiable validation.

![medical imaging](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2Feab1a2dc98561affced2f3a84976ed41%2FSIIM%202021%20Call%20For%20Models%20-%20Example.gif?generation=1620776949540106&alt=media)

This challenge is brought to life thanks to the NSF Convergence Accelerator Grant, on which SIIM is a Co-Principal Investigator. One of the main deliverables for Phase 1 of the grant was to create a prototype of a Model Zoo, that would allow for search, discovery, sharing, and 3rd party validation of AI models in medical imaging.

To populate the Model Zoo, SIIM has issued its first-ever Call for AI Models - the winning solutions will be presented at the [SIIM21 Annual Meeting](https://siim.org/page/siim2021) on May 24-27.

In addition, SIIM has partnered with FISABIO and RSNA, as well as HP and Intel who are providing $100,000 in prizes to organize this competition, in hopes to receive as many open source models as possible to add to the Model Zoo.

If SIIM + Partners are granted Phase 2 of this NSF Convergence Accelerator, the primary deliverable will be clinical research testing of collaborative model-centric AI platform to meet the urgent needs of scalable validation and translation of model-centric AI in medical imaging. Thus, the more verified models we are able to add to the Model Zoo, the better it is for the advancement of science and betterment of healthcare.

In the spirit of advancing research on this topic, we invite you to contribute your open sourced models for this competition to be included in the Model Zoo. This offer is extended to any participant. **Submit your open source model via the [Strait I3 Platform](https://straiti3.org/blog/index.php/welcome-how-can-we-help/)**.

## Acknowledgments

Thank you to the following supporters without whom this challenge would not have been possible:

### Accelerator Grant

![NSF Convergence Accelerator logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F45763df597489f278765c657a1c45fb5%2Fimage2.png?generation=1619563923915334&alt=media)

The National Science Foundation Convergence Accelerator for funding our grant proposal: Scalable, TRaceable Ai for ImagingTranslation: Innovation to Implementation for Accelerated Impact (STRAIT I3) that enabled organization of this Kaggle challenge.

---

### BIMCV-COVID19 Dataset Contributors

Thank you to FISABIO, BIMCV and Generalitat Valenciana for contributing The BIMCV-COVID19 Dataset to enable this competition:

![Fundacio Fisabio logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F32d140c4a76eac068bccf4a02f1a1265%2FFISABIO%20logo.JPG?generation=1620777164717857&alt=media)

The Foundation for the Promotion of Health and Biomedical Research of Valencia Region (FISABIO)
<http://fisabio.san.gva.es/en/fisabio>

![BIMCV Medical Imaging Databank of the Valencia Region logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F927b725379f44aa98a0ad115fa6975fd%2Fimage3.png?generation=1620277250014677&alt=media)

Regional Ministry of Health in Valencia Region-Spain
<https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/>

[http://www.san.gva.es](http://www.san.gva.es/)

![GENERALITAT VALENCIANA logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F59efd11a2b9e3037d696798562c57f6a%2Fimage5.png?generation=1620277590180558&alt=media)

Regional Ministry of Innovation, Universities, Science and Digital Society grant awarded through decree 51/2020 (Valencia-Spain)

[http://innova.gva.es](http://innova.gva.es/)

![RSNA Radiological Society of North America logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F4b4863b79bb2f75f44bbbf0471f3f351%2Fimage4.png?generation=1620928472546248&alt=media)

The Radiological Society of North America (RSNA) for contributing imaging data and annotations from the MIDRC-RICORD dataset for use in the challenge.

![THE CANCER IMAGING ARCHIVE logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2Fa3cd688800e528b1b3ce57afcacd0e31%2Fcancer-imaging-archive.png?generation=1622762569968740&alt=media)

The Cancer Imaging Archive originally published The MIDRC-RICORD Data.

---

### Sponsors

![Z HP intel logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F69fa7dcd45ca6300a1d1469cd529b720%2Fhp-intel-logos.png?generation=1620278542116703&alt=media)

HP and Intel, SIIM Corporate Impact Partners, for providing $100,000 in challenge prizes to support a first-ever call for open-source AI models to become part of a prototype Model Zoo built as a result of Phase 1 of the NSF Convergence Accelerator Grant. If SIIM and its partners are granted Phase 2, the prototype Model Zoo will be the basis for creating a clinical research testing of collaborative model-centric AI platform to meet the urgent needs of scalable validation and translation of model-centric AI in medical imaging.

---

### Data Annotators

![MD.ai logo](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F66c8e7dec8af76b607534077d11c10b1%2Fimage1.png?generation=1620278251747910&alt=media)

MD.ai for providing an annotation tool

We would also like to recognize the below group of amazing radiologists who contributed a considerable effort required to annotate the datasets in preparation for the challenge:

-   Paras Lakhani, MD, Thomas Jefferson University Hospital (Annotation Lead)
-   William Auffermann, MD, PhD, University of Utah Health Sciences Center
-   Diego Angulo, MD, San Juan de Alicante Hospital
-   Emi Benítez, MD, San Juan de Alicante Hospital
-   Peter Berquist, MD, Medstar Georgetown University Hospital
-   Tessa Cook, MD, PhD, CIIP, FSIIM, University of Pennsylvania
-   Maria Culiáñez, MD, San Juan de Alicante Hospital
-   Gustavo César de Antonio Corradi, MD, DASA
-   Suely Fazio Ferraciolli, MD, HCFMUSP
-   Joaquin Galant Herrero, MD, San Juan de Alicante Hospital
-   Lara Jaques, MD, San Juan de Alicante Hospital
-   Sarah Kamel, MD, Thomas Jefferson University Hospital
-   Alba Mas, MD, San Juan de Alicante Hospital
-   María Panadero, MD, San Juan de Alicante Hospital
-   Maansi Parekh, MD, Thomas Jefferson University Hospital
-   Michael Peterson, MD, University of Utah Health Sciences Center
-   Theresa Pham, MD, University of Utah Health Sciences Center
-   Prasanth Prasanna, MD, University of Utah Health Sciences Center
-   Mariola Sánchez, MD, San Juan de Alicante Hospital
-   Marcelo Straus Takahashi, MD, DASA
-   Marta Vidal, MD, San Juan de Alicante Hospital
-   Spencer Workman, MD, Vanderbilt University Medical Center

* * * * *

### Challenge Organizing Team

In addition, special thanks go out to **Katherine Andriole, PhD, FSIIM** and **John Mongan, MD, PhD**, representing the RSNA Machine Learning Steering Subcommittee, who were instrumental in helping to design, organize, and conduct the challenge.

Finally, huge thanks to the Co-chairs of the SIIM Machine Learning Committee, **Steve Langer, PhD, CIIP, FSIIM** and **George Shih, MD, MS**, as well as Co-chair of the SIIM Machine Learning Tools & Research Sub-committee **Bennett Landman, PhD**, and a Member of that Sub-committee, **Paras Lakhani, MD**, who were instrumental in designing, organizing, and conducting the challenge in collaboration with Kaggle, FISABIO, and the RSNA.

## Partners: HP & Intel

### SIIM Impact Partners: HP & Intel

![Intel banner](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2Fb71d285694b97160b63179988eea45e1%2FZ%20WW%20_%20Web%20Banner-2349x900px-Final.jpg?generation=1620279077936215&alt=media)

#### About Intel:

Intel Corporation is a world leader in the design and manufacturing of products and technologies that enrich the lives of every person on earth. Intel engineers solutions for customers' greatest challenges with reliable, cloud to edge computing, inspired by Moore's Law. Workstations powered by Intel technologies improve workflow and accelerate data science tasks by handling massive datasets while minimizing latency. Find out more: <https://www.intel.com/content/www/us/en/products/devices-systems/workstations.html>

#### About Z by HP:

HP Inc. creates technology that makes life better for everyone, everywhere. Through our portfolio of printers, PCs, 3D Printing Technologies, mobile devices, solutions, and services, we engineer experiences that amaze and enable workflows that are seamless. Z by HP workstations are designed to meet the demands of advanced analytics---right out of the box. Select ZBooks and Z desktops are preloaded and tested with the most popular data science tools to take the guess work out of creating the optimal data science computing environment. More information about Z by HP Data Science solutions is available at: <https://www.hp.com/datascience>

#### Why Intel & Z by HP for Data Science?:

Z by HP Data Science solutions powered by Intel technologies improve workflows and accelerate data science tasks by handling massive datasets while minimizing latency. Our solutions accelerate data science workflows and enhance your cloud experience, while allowing seamless access to critical data science tools.\
By using Z by HP & Intel Data Science Workstations you can:

-   Help achieve time savings and get business insights faster by analyzing and training larger datasets.
-   Reduce your operating costs by using pre-loaded tools and technology for your specific data science workflows.
-   Crush your next Kaggle competition or notebook using solutions built with your data journey in mind!

#### Learn more:

[Z by HP Brochure](https://h20195.www2.hp.com/v2/getpdf.aspx/4AA7-5783ENW.pdf)

[Intel Brochure](https://storage.googleapis.com/kaggle-media/competitions/SIIM2021/Intel_AI-workstations-for-data-scientists-BRIEF.pdf)

#### Z by HP & Intel Student Team Prize:

Z by HP and Intel are committed to supporting students in their pursuit of careers in data science and AI. Together, we are offering the top student-led team a brand-new data science ready Z by HP ZBook Studio Mobile Workstation with HP's pre-loaded software stack!

Review the [Prizes page](https://www.kaggle.com/c/siim-covid19-detection/overview/prizes) for details on the Student Team Prize!

#### Acknowledgments:

Z by HP and Intel are proud to support SIIM through our corporate partnership and specific collaboration with this Kaggle challenge. Thank you to the entire SIIM team and their multiple partners, RSNA, FISABIO, and BIMCV for their time and energy in putting together this challenge for the Kaggle community.

## Citation

Andrew Kemp, Anna Zawacki, Chris Carr, George Shih, John Mongan, Julia Elliott, Kaiwen, ParasLakhani, Phil Culliton. (2021). SIIM-FISABIO-RSNA COVID-19 Detection. Kaggle. https://kaggle.com/competitions/siim-covid19-detection


# Dataset Description

In this competition, we are identifying and localizing COVID-19 abnormalities on chest radiographs. This is an object detection and classification problem.

For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "none 1 0 0 1 1" ("none" is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

Further, for each test study, you should make a determination within the following labels:

```
'Negative for Pneumonia'
'Typical Appearance'
'Indeterminate Appearance'
'Atypical Appearance'
```

To make a prediction of one of the above labels, create a prediction string similar to the "none" class above: e.g. `atypical 1 0 0 1 1`

Please see the Evaluation page for more details about formatting predictions.

The images are in DICOM format, which means they contain additional data that might be useful for visualizing and classifying.

### Dataset information

The train dataset comprises 6,334 chest scans in DICOM format, which were de-identified to protect patient privacy.

All images were labeled by a panel of experienced radiologists for the presence of opacities as well as overall appearance.

Note that all images are stored in paths with the form `study`/`series`/`image`. The `study` ID here relates directly to the study-level predictions, and the `image` ID is the ID used for image-level predictions.

The hidden test dataset is of roughly the same scale as the training dataset.

### Files

-   **train_study_level.csv** - the train study-level metadata, with one row for each study, including correct labels.
-   **train_image_level.csv** - the train image-level metadata, with one row for each image, including both correct labels and any bounding boxes in a dictionary format. Some images in both test and train have multiple bounding boxes.
-   **sample_submission.csv** - a sample submission file containing all image- and study-level IDs.

### Columns

**train_study_level.csv**

-   `id` - unique study identifier
-   `Negative for Pneumonia` - `1` if the study is negative for pneumonia, `0` otherwise
-   `Typical Appearance` - `1` if the study has this appearance, `0` otherwise
-   `Indeterminate Appearance`  - `1` if the study has this appearance, `0` otherwise
-   `Atypical Appearance`  - `1` if the study has this appearance, `0` otherwise

**train_image_level.csv**

-   `id` - unique image identifier
-   `boxes` - bounding boxes in easily-readable dictionary format
-   `label` - the correct prediction label for the provided bounding boxes

### Citation

The **BIMCV-COVID19 Data** used by this challenge were originally published by the Medical Imaging Databank of the Valencia Region (BIMCV) in cooperation with The Foundation for the Promotion of Health and Biomedical Research of Valencia Region (FISABIO), and the Regional Ministry of Innovation, Universities, Science and Digital Society (Generalitat Valenciana), however the images were completely re-annotated using different annotation types. Users of this data must abide by the [BIMCV-COVID19 Dataset research Use Agreement](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/bimcv-covid19-dataset-research-use-agreement-2/). *Paper Reference: [BIMCV COVID-19+: a large annotated dataset of RX and CT images from COVID-19 patients](https://arxiv.org/abs/2006.01174)*

The **MIDRC-RICORD Data** used by this challenge were originally published by The Cancer Imaging Archive. The images were re-annotated for this challenge using a different annotation schema. Users of this data must abide by the [TCIA Data Usage Policy](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions) and the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) under which it has been published. Attribution should include references to citations listed on the [TCIA citation information page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281#702302814dc5f53338634b35a3500cbed18472e0) (page bottom). *Paper Reference: [The RSNA International COVID-19 Open Radiology Database (RICORD)](https://pubs.rsna.org/doi/full/10.1148/radiol.2021203957)*

Want to learn more about the curation, annotation methodology and characteristics of the dataset used in this challenge?

**Read The 2021 SIIM-FISABIO-RSNA Machine Learning COVID-19 Challenge: Annotation and Standard Exam Classification of COVID-19 Chest Radiographs.**

<https://doi.org/10.31219/osf.io/532ek>

To cite:

`1. Lakhani P, Mongan J, Singhal C, Zhou Q, Andriole KP, Auffermann WF, Prasanna P, Pham T, Peterson M, Bergquist PJ, Cook TS, Ferraciolli SF, de Antonio Corradi GC, Takahashi M, Workman SS, Parekh M, Kamel S, Galant JH, Mas-Sanchez A, Benítez EC, Sánchez-Valverde M, Jaques L, Panadero M, Vidal M, Culiáñez-Casas M, Angulo-Gonzalez DM, Langer SG, de la Iglesia Vaya M, Shih G. The 2021 SIIM-FISABIO-RSNA Machine Learning COVID-19 Challenge: Annotation and Standard Exam Classification of COVID-19 Chest Radiographs. [Internet]. OSF Preprints; 2021. Available from: osf.io/532ek`