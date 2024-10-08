## **Description**

**“Are we alone in the Universe?”**

It's one of the most profound—and perennial—human questions. As technology improves, we're finding new and more powerful ways to seek answers. The Breakthrough Listen team at the University of California, Berkeley, employs the world's most powerful telescopes to scan millions of stars for signs of technology. Now it wants the Kaggle community to help interpret the signals they pick up.

The Listen team is part of the Search for ExtraTerrestrial Intelligence (SETI) and uses the largest steerable dish on the planet, the 100-meter diameter Green Bank Telescope. Like any SETI search, the motivation to communicate is also the major challenge. Humans have built enormous numbers of radio devices. It's hard to search for a faint needle of alien transmission in the huge haystack of detections from modern technology.

Current methods use two filters to search through the haystack. First, the Listen team intersperses scans of the target stars with scans of other regions of sky. Any signal that appears in both sets of scans probably isn't coming from the direction of the target star. Second, the pipeline discards signals that don't change their frequency, because this means that they are probably nearby the telescope. A source in motion should have a signal that suggests movement, similar to the change in pitch of a passing fire truck siren. These two filters are quite effective, but we know they can be improved. The pipeline undoubtedly misses interesting signals, particularly those with complex time or frequency structure, and those in regions of the spectrum with lots of interference.

In this competition, use your data science skills to help identify anomalous signals in scans of Breakthrough Listen targets. Because there are no confirmed examples of alien signals to use to train machine learning algorithms, the team included some simulated signals (that they call “needles”) in the haystack of data from the telescope. They have identified some of the hidden needles so that you can train your model to find more. The data consist of two-dimensional arrays, so there may be approaches from computer vision that are promising, as well as digital signal processing, anomaly detection, and more. The algorithm that's successful at identifying the most needles will win a cash prize, but also has the potential to help answer one of the biggest questions in science.

![DSC_4014-Edit_2.jpg](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/DSC_4014-Edit_2.jpg)

### **Acknowledgments**

The Breakthrough Listen science and engineering effort is headquartered at the University of California, Berkeley SETI Research Center. The Breakthrough Prize Foundation funds the Breakthrough Initiatives which manages Breakthrough Listen. The Green Bank Observatory is supported by the National Science Foundation, and is operated by Associated Universities, Inc. under a cooperative agreement.

## **Evaluation**

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

### **Submission File**

For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
00034abb3629,0.5
0004be0baf70,0.5
0005be4d0752,0.5
etc.

```

## **Timeline**

- **May 10, 2021** - Start Date.
- **Aug 11, 2021** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **Aug 11, 2021** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **Aug 18, 2021** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## **Prizes**

### **Cash Prizes**

- 1st Place - $6,000
- 2nd Place - $5,000
- 3rd Place - $4,000

### **Additional Opportunities**

Thank you for joining our efforts! If you felt that you made innovative contributions in this competition we want to invite you to submit your work to present at an upcoming conference. Any participant, not necessarily with the highest score, is welcome to submit model (code) and documentation to a panel with international experts in SETI, radio astronomy, and machine learning. The panel will select the contribution submissions with the most original approach with respect to traditional SETI strategies. We will provide a limited budget for travel to the conference to share your experience with our community. An entry form for those of you interested in this phase will be posted in a discussion thread before the end of the competition.

Selected entrants can attend one of the upcoming conferences:

- Invitations (at least one) to Breakthrough Discuss, April 2022 in the San Francisco Bay Area.
- Invitations (at least one) to the International Astronautical Congress, September 2022 in Paris, France.

Alternatively selected entrants may visit the Breakthrough Listen team at UC Berkeley for discussions and collaboration at a mutually convenient time.

## **Data Information**

The Breakthrough Listen instrument at the Green Bank Telescope (GBT) is a digital spectrometer, which takes incoming raw data from the telescope (amounting to hundreds of TB per day) and performs a Fourier Transform to generate a spectrogram. These spectrograms, also referred to as filterbank files, or dynamic spectra, consist of measurements of signal intensity as a function of frequency and time.

Below is an example of an FM radio signal. This is not from the GBT, but from a small antenna attached to a software defined radio dongle (a $20 piece of kit that you can plug into your laptop to pick up signals). The data we get from the GBT are very similar, but split into larger numbers of frequency channels, covering a much broader instantaneous frequency range, and with much better sensitivity.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/b59a57f3-11a7-4493-8268-55c3fa632f7e/Untitled.png)

The screenshot above shows frequency on the horizontal axis (running from around 88.2 to 89.8 MHz) and time on the vertical axis. The bright orange feature at 88.5 MHz is the FM signal from KQED, a radio station in the San Francisco Bay Area. The solid yellow blocks on either side (one highlighted by the pointer in the screenshot) are the KQED “HD radio” signal (the same data as the FM signal, but encoded digitally). Additional FM stations are visible at different frequencies, including another obvious FM signal (without the corresponding digital sidebands) at 89.5 MHz.

Breakthrough Listen generates similar spectrograms to the one shown above, but typically spanning several GHz of the radio spectrum (rather than the approx. 2 MHz shown above). The data are stored either as filterbank format or HDF5 format files, but essentially are arrays of intensity as a function of frequency and time, accompanied by headers containing metadata such as the direction the telescope was pointed in, the frequency scale, and so on. We generate over 1 PB of spectrograms per year; individual filterbank files can be tens of GB in size. For the purposes of the Kaggle challenge, we have discarded the majority of the metadata and are simply presenting numpy arrays consisting of small regions of the spectrograms that we refer to as “snippets”.

Breakthrough Listen is searching for candidate signatures of extraterrestrial technology - so-called technosignatures. The main obstacle to doing so is that our own human technology (not just radio stations, but wifi routers, cellphones, and even electronics that are not deliberately designed to transmit radio signals) also gives off radio signals. We refer to these human-generated signals as “radio frequency interference”, or RFI.

One method we use to isolate candidate technosignatures from RFI is to look for signals that appear to be coming from particular positions on the sky. Typically we do this by alternating observations of our primary target star with observations of three nearby stars: 5 minutes on star “A”, then 5 minutes on star “B”, then back to star “A” for 5 minutes, then “C”, then back to “A”, then finishing with 5 minutes on star “D”. One set of six observations (ABACAD) is referred to as a “cadence”. Since we're just giving you a small range of frequencies for each cadence, we refer to the datasets you'll be analyzing as “cadence snippets”.

So, you want to see an example of an extraterrestrial signal? Here you are:

![Screen Shot 2021-05-03 at 11.39.42.png](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.39.42.png)

As the plot title suggests, this is the Voyager 1 spacecraft. Even though it's 20 billion kilometers from Earth, it's picked up clearly by the GBT. The first, third, and fifth panels are the “A” target (the spacecraft, in this case). The yellow diagonal line is the radio signal coming from Voyager. It's detected when we point at the spacecraft, and it disappears when we point away. It's a diagonal line in this plot because the relative motion of the Earth and the spacecraft imparts a Doppler drift, causing the frequency to change over time. As it happens, that's another possible way to reject RFI, which has a higher tendency to remain at a fixed frequency over time.

While it would be nice to train our algorithms entirely on observations of interplanetary spacecraft, there are not many examples of them, and we also want to be able to find a wider range of signal types. So we've turned to simulating technosignature candidates.

We've taken tens of thousands of cadence snippets, which we're calling the haystack, and we've hidden needles among them. Some of these needles look similar to the Voyager 1 signal above and should be easy to detect, even with classical detection algorithms. Others are hidden in noisy regions of the spectrum and will be harder, even though they might be relatively obvious on visual inspection:

![Screen Shot 2021-05-03 at 11.34.06.png](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.34.06.png)

After we perform the signal injections, we normalize each snippet, so you probably can't identify most of the needles just by looking for excess energy in the corresponding array. You'll likely need a more subtle algorithm that looks for patterns that appear only in the on-target observations.

Not all of the “needle” signals look like diagonal lines, and they may not be present for the entirety of all three “A” observations, but what they do have in common is that they are only present in some or all of the “A” observations (panels 1, 3, and 5 in the cadence snippets). Your challenge is to train an algorithm to find as many needles as you can, while minimizing the number of false positives from the haystack.

## **Citation**

Andrew Siemion, Don Diego Alonso, Walter Reade, Shirley Wang, Steve Croft, Yuhong Chen. (2021). SETI Breakthrough Listen - E.T. Signal Search. Kaggle. https://kaggle.com/competitions/seti-breakthrough-listen

# Data

## **Dataset Description**

In this competition you are tasked with looking for technosignature signals in cadence snippets taken from the Green Bank Telescope (GBT). Please read the extended description on the [Data Information](https://www.kaggle.com/c/seti-breakthrough-listen/overview/data-information) tab for detailed information about the data (that's too lengthy to include here).

## **Files**

- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has dimension `(6, 273, 256)`, with the 1st dimension representing the 6 positions of the cadence, and the 2nd and 3rd dimensions representing the 2D spectrogram.
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
- **old_leaky_data** - full pre-relaunch data, including test labels; you should not assume this data is helpful (it may or may not be).