# Overview

## Description

What if scientists could anticipate volcanic eruptions as they predict the weather? While determining rain or shine days in advance is more difficult, weather reports become more accurate on shorter time scales. A similar approach with volcanoes could make a big impact. Just one unforeseen eruption can result in tens of thousands of lives lost. If scientists could reliably predict when a volcano will next erupt, evacuations could be more timely and the damage mitigated.

Currently, scientists often identify “time to eruption” by surveying volcanic tremors from seismic signals. In some volcanoes, this intensifies as volcanoes awaken and prepare to erupt. Unfortunately, patterns of seismicity are difficult to interpret. In very active volcanoes, current approaches predict eruptions some minutes in advance, but they usually fail at longer-term predictions.

Enter Italy's Istituto Nazionale di Geofisica e Vulcanologia (INGV), with its focus on geophysics and volcanology. The INGV's main objective is to contribute to the understanding of the Earth's system while mitigating the associated risks. Tasked with the 24-hour monitoring of seismicity and active volcano activity across the country, the INGV seeks to find the earliest detectable precursors that provide information about the timing of future volcanic eruptions.

In this competition, using your data science skills, you’ll predict when a volcano's next eruption will occur. You'll analyze a large geophysical dataset collected by sensors deployed on active volcanoes. If successful, your algorithms will identify signatures in seismic waveforms that characterize the development of an eruption.

With enough notice, areas around a volcano can be safely evacuated prior to their destruction. Seismic activity is a good indicator of an impending eruption, but earlier precursors must be identified to improve longer-term predictability. The impact of your participation could be felt worldwide with tens of thousands of lives saved by more predictable volcanic ruptures and earlier evacuations.

## Evaluation

Submissions are evaluated on the [mean absolute error (MAE)](https://www.kaggle.com/wiki/MeanAbsoluteError) between the predicted loss and the actual loss.

### Submission File

For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Timeline

- Entry deadline: Same as the Final Submission Deadline
- Team Merger deadline: Same as the Final Submission Deadline
- **Final submission deadline: Jan 6, 2021**

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- **1st Place** – Kaggle Swag Prize
- **2nd Place** – Kaggle Swag Prize
- **3rd Place** – Kaggle Swag Prize

**Kaggle Swag Prizes** include the winner's choice of:

- Kaggle t-shirt
- Kaggle coffee mug
- Kaggle water bottle

## Citation

Andrea Cannata, ConcettoSpampinato, Flavio Cannavo, IsaakKavasidis, Maggie, Sohier Dane. (2020). INGV - Volcanic Eruption Prediction. Kaggle. https://kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe

# Data

## Dataset Description

Detecting volcanic eruptions before they happen is an important problem that has historically proven to be a very difficult. This competition provides you with readings from several seismic sensors around a volcano and challenges you to estimate how long it will be until the next eruption. The data represent a classic signal processing setup that has resisted traditional methods.

Identifying the exact sensors may be possible but would not be in the spirit of the competition nor further the scientific objectives. Please respect the importance of the problem and the time invested by the researchers at INGV in making this problem available by not seeking more metadata or information that would be unavailable in a real prediction context.

### Files

**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.