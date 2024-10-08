# Overview

## Description

## Goal of the Competition

The goal of this competition is to predict a neutrino particle's direction. You will develop a model based on data from the "IceCube" detector, which observes the cosmos from deep within the South Pole ice.

Your work could help scientists better understand exploding stars, gamma-ray bursts, and cataclysmic phenomena involving black holes, neutron stars and the fundamental properties of the neutrino itself.

## Context

One of the most abundant particles in the universe is the neutrino. While similar to an electron, the nearly massless and electrically neutral neutrinos have fundamental properties that make them difficult to detect. Yet, to gather enough information to probe the most violent astrophysical sources, scientists must estimate the direction of neutrino events. If algorithms could be made considerably faster and more accurate, it would allow for more neutrino events to be analyzed, possibly even in real-time and dramatically increase the chance to identify cosmic neutrino sources. Rapid detection could enable networks of telescopes worldwide to search for more transient phenomena.

Researchers have developed multiple approaches over the past ten years to reconstruct neutrino events. However, problems arise as existing solutions are far from perfect. They're either fast but inaccurate or more accurate at the price of huge computational costs.

The IceCube Neutrino Observatory is the first detector of its kind, encompassing a cubic kilometer of ice and designed to search for the nearly massless neutrinos. An international group of scientists is responsible for the scientific research that makes up the IceCube Collaboration.

By making the process faster and more precise, you'll help improve the reconstruction of neutrinos. As a result, we could gain a clearer image of our universe.

![IceCube South Pole Neutrino Observatory](https://storage.googleapis.com/kaggle-media/competitions/IceCube/icecube_detector.jpg)

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/icecube-neutrinos-in-deep-ice/overview/code-requirements) for details.

## Evaluation

Submissions are evaluated using the mean angular error between the predicted and true event origins. See [this notebook](https://www.kaggle.com/code/sohier/mean-angular-error) for a python copy of the metric.

Submission File
---------------

For each `event_id` in the test set, you must predict the `azimuth` and `zenith`. The file should contain a header and have the following format:

```
event_id,azimuth,zenith
730,1,1
769,1,1
774,1,1
etc.
```

## Timeline

- January 19, 2023 - Start Date.

- April 12, 2023 - Entry Deadline. You must accept the competition rules before this date in order to compete.

- April 12, 2023 - Team Merger Deadline. This is the last day participants may join or merge teams.

- April 19, 2023 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

Leaderboard Prizes
------------------

\$40,000 will be awarded to the Top 3 teams with the highest scores on the Kaggle Leaderboard at the conclusion of the competition:

- 1st Place - \$18,000
- 2nd Place - \$12,000
- 3rd Place - \$10,000

"Most Interesting Solution Writeup" Prizes
------------------------------------------

\$5,000 will be awarded to 5 teams (\$1,000 each) for the "most interesting solution writeup" from the top 30 final leaderboard finishers at the conclusion of the competition. Please see ["Writeup Scoring Rubrics"](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/overview/writeup-scoring-rubrics) for the details of the evaluation criteria.

As a condition to being awarded a Prize, a Prize winner must provide a detailed write-up on their solution in the competition forums within 14 days of the conclusion of the competition, i.e. May 3, 2023 11:59PM UTC.

Early Sharing Prize
-------------------

\$5,000 will be awarded to the team with the highest private leaderboard score as of February 2, 2023 11:59PM UTC (2 weeks from the start of the competition) among submission notebooks published by February 4, 2023 11:59PM UTC. Please see ["Early Sharing Prize"](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/overview/early-sharing-prize) for the details on how the Early Sharing Prize will be awarded at the conclusion of the competition.

## Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

## This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9` hours run-time
- GPU Notebook `<= 9` hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Writeup Scoring Rubrics

| Criterion | Description | Weight |
| --- | --- | --- |
| Technical Details & Reproducibility | Is the code well documented and the description well written? Is the documentation sufficient for training and deploying the model from scratch? | 30% |
| Innovation | Is the full model, or certain parts of the approach particularly innovative? Are certain non-standard methods used or new methods invented? What goes beyond standard ML toolchains? | 20% |
| Generalness | Is the method general enough such that it could be applied to other neutrino telescopes that feature a similar detector concept but a different arrangement of a different number of modules? For example, the detectors of <https://www.km3net.org/> that are currently under construction. | 20% |
| Lessons Learned | What were the exploratory steps taken on the way from the first try to the final model? What approaches did not work well? | 10% |
| Robustness | How robust is the method to perturbations in the data? For example, change an event by adding random numbers taken from a normal distribution with standard deviation 5 ns to each pulse time. Are the predicted neutrino angles stable under such changes? | 10% |
| Explanation | Can the method be visualized in some way and can the parts of the model be motivated/explained as why these are necessary and what exactly they do? | 10% |

## Early Sharing Prize


We are awarding a \$5,000 cash prize to encourage participants to share information earlier and help the community make more progress over the course of the competition.

To be eligible for the Early Sharing Prize, you will need to:

-   Submit one or more notebooks before February 2, 2023 11:59PM UTC and publish them before February 4, 2023 11:59PM UTC. The delayed publication date is intended to allow you to participate in the early sharing prize without being concerned that someone else might re-submit your work.
-   Keep the notebooks and any datasets it uses publicly available until the prize is awarded at the end of the competition.

To avoid creating a disincentive for teaming up, the prize will be awarded to the members of the team as it exists at 2 weeks from the start of the competition.

This is an experimental prize and we would appreciate feedback on the concept.

## Additional Resources

Here are a few introductory videos:

- [Neutrino, measuring the unexpected--IceCube](https://www.youtube.com/watch?v=iv-Rz3-s4BM)
- [Chasing the Ghost Particle: From the South Pole to the Edge of the Universe](https://www.youtube.com/watch?v=xuyTgAlPOGY)
- [Uncharted Cosmos: Mapping the Universe with IceCube](https://www.youtube.com/watch?v=GBPNe_0WY4Y)

Below is a collection of useful references on the topic:

## Scientific Results (selected highlights):

- [Evidence for neutrino emission from the nearby active galaxy NGC 1068](https://icecube.wisc.edu/news/press-releases/2022/11/icecube-neutrinos-give-us-first-glimpse-into-the-inner-depths-of-an-active-galaxy/)
- [Multi-messenger observations of a flaring blazar coincident with high-energy neutrino IceCube-170922A](https://arxiv.org/abs/1807.08816)
- [Neutrino emission from the direction of the blazar TXS 0506+056 prior to the IceCube-170922A alert](https://arxiv.org/abs/1807.08794)
- [Observation of High-Energy Astrophysical Neutrinos in Three Years of IceCube Data](https://arxiv.org/abs/1405.5303)
- [Detection of a particle shower at the Glashow resonance with IceCube](https://arxiv.org/abs/2110.15051)
- [Measurement of Atmospheric Neutrino Oscillations at 6--56 GeV with IceCube DeepCore](https://arxiv.org/abs/1707.07081)
- [Time-integrated Neutrino Source Searches with 10 years of IceCube Data](https://arxiv.org/abs/1910.08488)
- [Measurement of the multi-TeV neutrino cross section with IceCube using Earth absorption](https://arxiv.org/abs/1711.08119)
- [An eV-scale sterile neutrino search using eight years of atmospheric muon neutrino data from the IceCube Neutrino Observatory](https://arxiv.org/abs/2005.12942)

## Detector & Ice:

- [The IceCube Neutrino Observatory: Instrumentation and Online Systems](https://arxiv.org/abs/1612.05093)
- [First Year Performance of The IceCube Neutrino Telescope](https://arxiv.org/abs/astro-ph/0604450)
- [The IceCube Data Acquisition System: Signal Capture, Digitization, and Timestamping](https://arxiv.org/abs/0810.4930)
- [The Design and Performance of IceCube DeepCore](https://arxiv.org/abs/1109.6096)
- [Measurement of South Pole ice transparency with the IceCube LED calibration system](https://arxiv.org/abs/1301.5361)

## Event Reconstruction:

- [Energy Reconstruction Methods in the IceCube Neutrino Telescope](https://arxiv.org/abs/1311.4767)
- [Low Energy Event Reconstruction in IceCube DeepCore](https://arxiv.org/abs/2203.02303)
- [A flexible event reconstruction based on machine learning and likelihood principles](https://arxiv.org/abs/2208.10166)
- [Graph Neural Networks for Low-Energy Event Classification & Reconstruction in IceCube](https://arxiv.org/abs/2209.03042)
- [A Fast Algorithm for Muon Track Reconstruction and its Application to the ANTARES Neutrino Telescope](https://arxiv.org/abs/1105.4116)
- [An algorithm for the reconstruction of neutrino-induced showers in the ANTARES neutrino telescope](https://arxiv.org/abs/1708.03649)
- [Muon Track Reconstruction and Data Selection Techniques in AMANDA](https://arxiv.org/abs/astro-ph/0407044)
- [A Convolutional Neural Network based Cascade Reconstruction for the IceCube Neutrino Observatory](https://arxiv.org/abs/2101.11589)
- [Event reconstruction for KM3NeT/ORCA using convolutional neural networks](https://arxiv.org/abs/2004.08254)
- [An improved method for measuring muon energy using the truncated mean of dE/dx](https://arxiv.org/abs/1208.3430)
- [A muon-track reconstruction exploiting stochastic losses for large-scale Cherenkov detectors](https://arxiv.org/abs/2103.16931)
- [Improvement in Fast Particle Track Reconstruction with Robust Statistics](https://arxiv.org/abs/1308.5501)

## Other Neutrino Telescopes (past, present & future):

- [The AMANDA Neutrino Telescope: Principle of Operation and First Results](https://arxiv.org/abs/astro-ph/9906203)
- [ANTARES: the first undersea neutrino telescope](https://arxiv.org/abs/1104.1607)
- [Letter of Intent for KM3NeT 2.0](https://arxiv.org/abs/1601.07459)
- [IceCube-Gen2: The Window to the Extreme Universe](https://arxiv.org/abs/2008.04323)
- [Letter of Intent: The Precision IceCube Next Generation Upgrade (PINGU)](https://arxiv.org/abs/1401.2046)
- [Baikal-GVD: status and prospects](https://arxiv.org/abs/1808.10353)
- [The Pacific Ocean Neutrino Experiment](https://arxiv.org/abs/2005.09493)

## Acknowledgements


|  |  |
| --- | --- |
| ![](https://storage.googleapis.com/kaggle-media/competitions/IceCube/icecube_logo_large1.png) | The [IceCube Neutrino Observatory](https://icecube.wisc.edu/) is the world's largest neutrino detector and is providing the data for this challenge as well as helping with the organization. |
| ![](https://storage.googleapis.com/kaggle-media/competitions/IceCube/TUM_Logo_blau_rgb_p1.png) | The [Technical University of Munich](https://www.tum.de/), or short TUM, is one of the top universities in Europe, a member of the IceCube collaboration, and the home of the competition organizers. |
| ![](https://storage.googleapis.com/kaggle-media/competitions/IceCube/MDSI.png) | The [Munich Data Science Institute](https://www.mdsi.tum.de/en/mdsi/home/) is TUM's central data science infrastructure and competence center. The MDSI is helping with the organization of this kaggle challenge, and supporting its realization. |
| ![](https://storage.googleapis.com/kaggle-media/competitions/IceCube/sfb_logo_farbe_trans.png) | The [Collaborative Research Center SFB 1258](https://www.sfb1258.de/) is a reserach collaboration focused on neutrinos and dark matter in astro- and particle physics, and a partner supporting this kaggle challenge. |
| ![](https://storage.googleapis.com/kaggle-media/competitions/IceCube/Logo_Origins_RGB.jpg) | The [Excellence Cluster ORIGINS](https://www.origins-cluster.de/) investigates the development of the Universe from the Big Bang to the emergence of life. The cluster and in particular its data science laboratory ODSL support this ML challenge. |
| ![](https://storage.googleapis.com/kaggle-media/competitions/IceCube/PUNCH4NFDI-Logo_RGB.png) | [PUNCH4NFDI](https://www.punch4nfdi.de/) is a large German research consortium under the [NFDI](https://www.nfdi.de/) infrastructure, focused on particle, astro-, astroparticle, hadron and nuclear physics, and is supporting this ML challenge. |

## Citation

Ashley Chow, Lukas Heinrich, Philipp Eller, Rasmus Ørsøe, Sohier Dane. (2023). IceCube - Neutrinos in Deep Ice. Kaggle. https://kaggle.com/competitions/icecube-neutrinos-in-deep-ice

# Dataset Description

The goal of this competition is to identify which direction neutrinos detected by the [IceCube neutrino observatory](https://icecube.wisc.edu/science/icecube/) came from. When detection events can be localized quickly enough, traditional telescopes are recruited to investigate short-lived neutrino sources such as supernovae or gamma ray bursts. Because the sky is huge better localization will not only associate neutrinos with sources but also to help partner observatories limit their search space. With an average of three thousand events per second to process, it's difficult to keep up with the stream of data using traditional methods. Your challenge in this competition is to quickly and accurately process a large number of events.

This competition uses a hidden test set. When your submitted notebook is scored the actual test data (including a full length sample submission) will be made available to your notebook. Expect to see roughly one million events in the hidden test set, split between multiple batches.

Files
-----

[train/test]_meta.parquet

-   `batch_id` (`int`): the ID of the batch the event was placed into.
-   `event_id` (`int`): the event ID.
-   `[first/last]_pulse_index` (`int`): index of the first/last row in the features dataframe belonging to this event.
-   `[azimuth/zenith]` (`float32`): the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. The target columns. Not provided for the test set. The direction vector represented by zenith and azimuth points to where the neutrino came from.
-   NB: Other quantities regarding the event, such as the interaction point in `x, y, z` (vertex position), the neutrino energy, or the interaction type and kinematics are not included in the dataset.

[train/test]/batch_[n].parquet Each batch contains tens of thousands of events. Each event may contain thousands of pulses, each of which is the digitized output from a photomultiplier tube and occupies one row.

-   `event_id` (`int`): the event ID. Saved as the index column in parquet.
-   `time` (`int`): the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.
-   `sensor_id` (`int`): the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.
-   `charge` (`float32`): An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has `float16` precision but is stored as `float32` due to limitations of the version of pyarrow the data was prepared with.
-   `auxiliary` (`bool`): If `True`, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If `False`, then this pulse was contributed to the trigger decision and the pulse was fully digitized.

sample_submission.parquet An example submission with the correct columns and properly ordered event IDs. The sample submission is provided in the parquet format so it can be read quickly but *your final submission must be a csv*.

`sensor_geometry.csv` The `x`, `y`, and `z` positions for each of the 5160 IceCube sensors. The row index corresponds to the `sensor_idx` feature of pulses. The `x`, `y`, and `z` coordinates are in units of meters, with the origin at the center of the IceCube detector. The coordinate system is right-handed, and the z-axis points upwards when standing at the South Pole. You can convert from these coordinates to `azimuth` and `zenith` with the following formulas (here the vector (x,y,z) is normalized):

```
x = cos(azimuth) * sin(zenith)
y = sin(azimuth) * sin(zenith)
z = cos(zenith)

```

Example Event
-------------

The following image shows a visual representation of the features of an IceCube event in the dataset. The colorful dots represent sensors that logged at least one pulse in the event. The size of the dots corresponds to the total charge of all pulses while the color indicates the time of the first pulse.\
The left panel shows only pulses with `auxiliary==False`, and the right panel with `auxiliary==True`. The small, gray points indicate the positions of all 5160 IceCube sensors. The red arrow shows the true neutrino direction of that event, i.e. the regression target.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1132983%2F6891ec67d9d40315637b1b292c3a486b%2FExample_event.png?generation=1666631264548536&alt=media)
