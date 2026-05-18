# Overview

## Overview

The goal of this competition is to detect and classify seizures and other types of harmful brain activity. You will develop a model trained on electroencephalography (EEG) signals recorded from critically ill hospital patients.

Your work may help rapidly improve electroencephalography pattern classification accuracy, unlocking transformative benefits for neurocritical care, epilepsy, and drug development. Advancement in this area may allow doctors and brain researchers to detect seizures or other brain damage to provide faster and more accurate treatments.

### Description

From stethoscopes to tongue depressors, doctors rely on many tools to treat their patients. Physicians use electroencephalography with critically ill patients to detect seizures and other types of brain activity that can cause brain damage. You can learn about how doctors interpret these EEG signals in these videos:\
EEG Talk - ACNS Critical Care EEG Terminology 2021 [(Part 1)](https://www.youtube.com/watch?v=S9NLrhj0x-M&t) [(Part 2)](https://www.youtube.com/watch?v=4D9R2WIKr-A) [(Part 3)](https://www.youtube.com/watch?v=-R5yUX7p_j4) [(Part 4)](https://www.youtube.com/watch?v=OknS2ObD9-g&t) [(Part 5)](https://www.youtube.com/watch?v=2c7ABQRkn3s)

Currently, EEG monitoring relies solely on manual analysis by specialized neurologists. While invaluable, this labor-intensive process is a major bottleneck. Not only can it be time-consuming, but manual review of EEG recordings is also expensive, prone to fatigue-related errors, and suffers from reliability issues between different reviewers, even when those reviewers are experts.

Competition host Sunstella Foundation was created in 2021 during the COVID pandemic to help minority graduate students in technology overcome challenges and celebrate their achievements. These students are vital to America's technology leadership and diversity. Through workshops, forums, and competitions, the Sunstella Foundation provides mentorship and career advice to support their success.

Sunstella Foundation is joined by Persyst, Jazz Pharmaceuticals, and the Clinical Data Animation Center (CDAC), whose research aims to help people preserve and enhance brain health.

Your work in automating EEG analysis will help doctors and brain researchers detect seizures and other types of brain activity that can cause brain damage, so that they can give treatments more quickly and accurately. The algorithms developed in this contest may also help researchers who are working to develop drugs to treat and prevent seizures.

There are six patterns of interest for this competition: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other". Detailed explanations of these patterns are [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

The EEG segments used in this competition have been annotated, or classified, by a group of experts. In some cases experts completely agree about the correct label. On other cases the experts disagree. We call segments where there are high levels of agreement "idealized" patterns. Cases where ~1/2 of experts give a label as "other" and ~1/2 give one of the remaining five labels, we call "proto patterns". Cases where experts are approximately split between 2 of the 5 named patterns, we call "edge cases".

Examples of EEG Patterns with Different Levels of Expert Agreement:\
![](https://storage.googleapis.com/kaggle-media/competitions/Harvard%20Medical%20School/eFig2.png)\
*Please refer to [Data tab](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data) for full screen PDF page of each subfigure.*

This figure shows selected examples of EEG patterns with different level of agreement. Rows are structured with the 1st row seizure, 2nd row LPDs, 3rd row GPDs, 4th row LRDA, and 5th row GRDA. Column-wise, examples of idealized forms of patterns are in the 1st column (A). These are patterns with uniform expert agreement. The 2nd column (B) are proto or partially formed patterns. About half of raters labeled these as one IIIC pattern and the other half labeled "Other". The 3rd and 4th columns (C, D) are edge cases (about half of raters labeled these one IIIC pattern and half labeled them as another IIIC pattern).

For B-1 there is rhythmic delta activity with some admixed sharp discharges within the 10 second raw EEG, and the spectrogram shows that this segment may belong to the tail end of a seizure, thus disagreement between SZ and "Other" makes sense. B-2 shows frontal lateralized sharp transients at ~1Hz, but they have a reversed polarity, suggesting they may be coming from a non-cerebral source, thus the split between LPD and "Other" (artifact) makes sense. B-3 has diffused semi-rhythmic delta background with poorly formed low amplitude generalized periodic discharges with s shifting morphology making it a proto-GPD type pattern. B-4 shows semi-rhythmic delta activity with unstable morphology over the right hemisphere, a proto-LRDA pattern. B-5 shows a few waves of rhythmic delta activity with an unstable morphology and is poorly sustained, a proto-GRDA. C-1 shows 2Hz LPDs showing an evolution with increasing amplitude evolving underlying rhythmic activity, a pattern between LPDs and the beginning of a seizure, an edge-case. D-1 shows abundant GPDs on top of a suppressed background with frequency of 1-2Hz. The average over the 10-seconds is close to 1.5Hz, suggesting a seizure, another edge case. C-2 is split between LPDs and GPDs. The amplitude of the periodic discharges is higher over the right, but a reflection is also seen on the left. D-2 is tied between LPDs and LRDA. It shares some features of both; in the temporal derivations it looks more rhythmic whereas in the parasagittal derivations it looks periodic. C-3 is split between GPDs and LRDA. The ascending limb of the delta waves have a sharp morphology, and these periodic discharges are seen on both sides. The rhythmic delta appears to be of higher amplitude over the left, but there is some reflection of the activity on the left. D-3 is split between GPDs and GRDA. The ascending limb of the delta wave has a sharp morphology and there is asymmetry in slope between ascending and descending limbs making it an edge case. C-4 is split between LRDA and seizure. It shows 2Hz LRDA on the left, and the spectrogram shows that this segment may belong to the tail end of a seizure, an edge-case. D-4 is split between LRDA and GRDA. The rhythmic delta appears to be of higher amplitude over the left, but there is some reflection of the activity on the right. C-5 is split between GRDA and seizure. It shows potentially evolving rhythmic delta activity with poorly formed embedded epileptiform discharges, a pattern between GRDA and seizure, an edge-case. D-5 is split between GRDA and LPDs. There is generalized rhythmic delta activity, while the activity on the right is somewhat higher amplitude and contains poorly formed epileptiform discharges suggestive of LPDs, an edge-case. Note: Recording regions of the EEG electrodes are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

### Evaluation

Submissions are evaluated on the [Kullback Liebler divergence](https://www.kaggle.com/code/metric/kullback-leibler-divergence/notebook) between the predicted probability and the observed target.

#### Submission File

For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

### Timeline

- **January 8, 2024** - Start Date.
- **April 1, 2024** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **April 1, 2024** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **April 8, 2024** - Final Submission Deadline.

Al deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

- 1st Place - \$20,000
- 2nd Place - \$12,000
- 3rd Place - \$7,000
- 4th Place - \$6,000
- 5th Place - \$5,000

### Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

**This is a Code Competition**

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9` hours run-time
- GPU Notebook `<= 9` hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

### Acknowledgements

![](https://storage.googleapis.com/kaggle-media/competitions/Harvard%20Medical%20School/CCEMRC-logo.gif)

We gratefully acknowledge the work of the experts from the [Critical Care EEG Monitoring Research Consortium](https://www.acns.org/research/critical-care-eeg-monitoring-research-consortium-ccemrc) (CCEMRC), who provided the labels for model training and evaluation, and to others who contributed to the scientific work that enabled this competition. Specifically, we acknowledge the contributions of (numbers refer to institutional affiliations in the list below):

Jin Jing^1,2^, Wendong Ge, Aaron F. Struck^3,4^ , Marta Bento Fernandes^1,2^, Shenda Hong^5^, Sungtae An^7^, Safoora Fatima^3^, Aline Herlopian^8^, Ioannis Karakis^9^, Jonathan J. Halford^10^, Marcus Ng^11^, Emily L. Johnson^12^, Brian Appavu^13^, Rani A. Sarkis^14^, Gamaleldin Osman^15^, Peter W. Kaplan^12^, Monica B. Dhakar^16^, Lakshman Arcot Jayagopal^17^, Zubeda Sheikh^18^, Olha Taraschenko^17^, Sarah Schmitt^10^, Hiba A. Haider^19^, Jennifer A. Kim^8^, Christa B. Swisher^20^, Nicolas Gaspard^21^, Mackenzie C. Cervenka^12^, Andres Rodriguez^9^, Jong Woo Lee^14^, Mohammad Tabaeizadeh^1,2^, Emily J. Gilmore^8^, Kristy Nordstrom^1^, Ji Yeoun Yoo^22^, Manisha Holmes^23^, Susan T. Herman^24^, Jennifer A. Williams^25^, Jay Pathmanathan^26^, Fábio A. Nascimento^1,2^, Ziwei Fan^1,2^, Samaneh Nasiri^1,2^, Mouhsin M. Shafi^27^, Sydney S. Cash^1,2^, Daniel B. Hoch^1,2^, Andrew J. Cole^1,2^, Eric S. Rosenthal^1,2^, Sahar F. Zafar^1,2^, Jimeng Sun^5^, M. Brandon Westover^1,2^

1-Massachusetts General Hospital/Harvard Medical School Department of Neurology, MA\
2-Massachusetts General Hospital Clinical Data Animation Center (CDAC), MA\
3-University of Wisconsin-Madison Department of Neurology\
4-William S Middleton Memorial Veterans Hospital Madison, WI\
5-National Institute of Health Data Science, Peking University, Beijing, China\
6-University of Illinois at Urbana-Champaign, College of Computing, Champaign, IL\
7-Georgia Institute of Technology, College of Computing, Atlanta, GA\
8-Yale University-Yale New Haven Hospital, CT\
9-Emory University School of Medicine, GA\
10-Medical University of South Carolina, SC\
11-University of Manitoba, Canada\
12-Johns Hopkins School of Medicine, MD\
13-University of Arizona College of Medicine, AZ\
14-Brigham and Women's Hospital, MA\
15-Mayo Clinic-Rochester, MN\
16-Warren Alpert School of Medicine of Brown University, Providence, RI\
17-University of Nebraska Medical Center, NE\
18-West Virginia University Hospitals, WV\
19-University of Chicago, Chicago, IL\
20-Atrium Health, NC\
21-Université Libre de Bruxelles - Hôpital Erasme, Belgium\
22-Icahn School of Medicine, Mount Sinai, NY\
23-New York University (NYU) Grossman School of Medicine, NY\
24-Barrow Neurological Institute, Phoenix, AZ\
25-Mater Misericordiae University Hospital, Dublin, Ireland.\
26-University of Pennsylvania, PA\
27-Beth Israel Deaconess Medical Center/Harvard Medical School, MA

### Citation

Jin Jing, Zhen Lin, Chaoqi Yang, Ashley Chow, Sohier Dane, Jimeng Sun, M. Brandon Westover. (2024). HMS - Harmful Brain Activity Classification . Kaggle. https://kaggle.com/competitions/hms-harmful-brain-activity-classification

# Data

## Dataset Description

The goal of this competition is to detect and classify seizures and other types of harmful brain activity in electroencephalography (EEG) data. Even experts find this to be a challenging task and often disagree about the correct labels.

This is a code competition. Only a few examples from the test set are available for download. When your submission is scored the test folders will be replaced with versions containing the complete test set.

### Files

**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.
