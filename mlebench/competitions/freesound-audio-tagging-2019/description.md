# Overview

## Description

One year ago, Freesound and Google's Machine Perception hosted an audio tagging competition challenging Kagglers to build a general-purpose auto tagging system. This year they're back and taking the challenge to the next level with multi-label audio tagging, doubled number of audio categories, and a *noisier than ever* training set. If you like raising your ML game, this challenge is for you.

![Freesound](https://storage.googleapis.com/kaggle-media/competitions/freesound/task2_freesound_audio_tagging.png)

Here's the background: Some sounds are distinct and instantly recognizable, like a baby's laugh or the strum of a guitar. Other sounds are difficult to pinpoint. If you close your eyes, could you tell the difference between the sound of a chainsaw and the sound of a blender?

Because of the vastness of sounds we experience, no reliable automatic general-purpose audio tagging systems exist. A significant amount of manual effort goes into tasks like annotating sound collections and providing captions for non-speech events in audiovisual content.

To tackle this problem, [Freesound](https://freesound.org/) (an initiative by [MTG-UPF](https://www.upf.edu/web/mtg) that maintains a collaborative database with over 400,000 Creative Commons Licensed sounds) and [Google Research's Machine Perception Team](https://research.google.com/teams/perception/) (creators of [AudioSet](https://research.google.com/audioset/), a large-scale dataset of manually annotated audio events with over 500 classes) have teamed up to develop the dataset for this new competition.

To win this competition, Kagglers will develop an algorithm to tag audio data automatically using a diverse vocabulary of 80 categories.

If successful, your systems could be used for several applications, ranging from automatic labelling of sound collections to the development of systems that automatically tag video content or recognize sound events happening in real time.

Ready to raise your game? Join the competition!

Note, this competition is similar in nature to [this competition ](https://www.kaggle.com/c/freesound-audio-tagging)with a new dataset, and multi-class labels.

## Organizers

- [Eduardo Fonseca](http://www.eduardofonseca.net/), [MTG-UPF](https://www.upf.edu/web/mtg), Barcelona
- [Manoj Plakal](https://ai.google/research/people/author8115), [Google's Sound Understanding](https://research.google.com/audioset////////about.html), New York
- [Frederic Font](https://ffont.github.io/), [MTG-UPF](https://www.upf.edu/web/mtg), Barcelona
- [Dan Ellis](https://ai.google/research/people/DanEllis), [Google's Sound Understanding](https://research.google.com/audioset////////about.html), New York

> This is a Kernels-only competition. Refer to [Kernels Requirements](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/kernels-requirements) for details.

## Evaluation

The task consists of predicting the audio labels (tags) for every test clip. Some test clips bear one label while others bear several labels. The predictions are to be done at the clip level, i.e., no start/end timestamps for the sound events are required.

The primary competition metric will be label-weighted [label-ranking average precision](https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision) (*lwlrap*, pronounced "Lol wrap"). This measures the average precision of retrieving a ranked list of relevant labels for each test clip (i.e., the system ranks all the available labels, then the precisions of the ranked lists down to each true label are averaged). This is a generalization of the mean reciprocal rank measure (used in last year's edition of the competition) for the case where there can be multiple true labels per test item. The novel "label-weighted" part means that the overall score is the average over all the *labels* in the test set, where each label receives equal weight (by contrast, plain *lrap* gives each *test item* equal weight, thereby discounting the contribution of individual labels when they appear on the same item as multiple other labels).

We use label weighting because it allows per-class values to be calculated, and still have the overall metric be expressed as simple average of the per-class metrics (weighted by each label's prior in the test set). For participant's convenience, a Python implementation of lwlrap is provided in this public [Google Colab](https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8).

## Submission File

For each `fname` in the test set, you must predict the probability of each label. The file should contain a header and have the following format:

```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

As we will be switching out test data to re-evaluate kernels on stage 2 data to populate the private leaderboard, submissions must be named submission.csv.

## Prizes

The following prizes will be awarded to the winners of the competition:

- 1st Place - $2,200
- 2nd Place - $1,200
- 3rd Place - $800
- [Judges' Award](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/judges-award) - $800

## Timeline

- June 3, 2019 11:59 PM UTC - Entry deadline. You must accept the competition rules before this date in order to compete.

- June 3, 2019 11:59 PM UTC - Team Merger deadline. This is the last day participants may join or merge teams.

- June 11, 2019 11:59 AM UTC - Final submission deadline.

The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Note, as this competition is a Kernels-only, two-stage competition, following the final submission deadline for the competition, your kernel code will be re-run on a privately-held test set that is not provided to you. It is your model's score against this private test set that will determine your ranking on the private leaderboard and final standing in the competition. The leaderboard will be updated in the days following the competition's completion, and our team will announce that the re-run has been completed and leaderboard finalized with an announcement made on the competition forums.

## DCASE

This competition is framed under the fifth edition of the [Detection and Classification of Acoustic Scenes and Events (DCASE) Challenge](http://dcase.community/challenge2019/). Building up on the success of the previous editions, DCASE 2019 Challenge continues to support the development of computational scene and event analysis methods by comparing different approaches using common publicly available datasets. Since DCASE may be new to some Kagglers, we provide some basic information next.

### DCASE Challenge

[DCASE 2019 Challenge](http://dcase.community/challenge2019/) features 5 tasks and Freesound Audio Tagging 2019 is [Task 2](http://dcase.community/challenge2019/task-audio-tagging). By participating in this competition, you are essentially participating in DCASE 2019 Task 2. However, in order to be considered for DCASE Task 2 records, participants must additionally submit a technical report and a *yaml* file describing the submitted system. The provided information will be published on the DCASE Challenge website (see [last year's](http://dcase.community/challenge2018/task-general-purpose-audio-tagging-results) as en example). We encourage participants to submit these additional files as it is a great opportunity to disseminate your work and it will help the community to better understand how the algorithm works. More info about how to submit these files through the DCASE submission system will be available soon.

### DCASE Workshop

[DCASE 2019 Workshop](http://dcase.community/workshop2019/) aims to provide a venue for researchers working on computational analysis of sound events and scene analysis to present and discuss their results. Results of the DCASE Challenge are typically presented as a part of the DCASE Workshop, which will be held on 25-26 October 2019, New York, US. To this end, participants must submit a research paper to the DCASE 2019 Workshop, the submission deadline being 12 July 2019 (few weeks after the competition closes). More info will be available in the Workshop paper instructions site.

Should you have any doubts, please contact eduardo.fonseca@upf.edu.

Relevant links:

- Quick glance on both DCASE 2019 Workshop and Challenge: <http://dcase.community/>
- Last year's DCASE Workshop: <http://dcase.community/workshop2018/>

## Judges Award

In addition to the prizes given to the top-3 scoring teams in the private leaderboard, a Judges' Award prize will also be given which is not necessarily based on the leaderboard scores. The goal of the Judges' Award is to encourage contestants to use novel and problem-specific approaches which leverage knowledge of the audio domain. Another key factor is computational efficiency: we want to promote models that can fit within reasonable resource constraints, as opposed to very large models that may merely memorize a dataset and may not fit within typical deployment environments on mobile devices, for example.

For the Judges' Award, submissions will be evaluated according to:

- Innovation and novelty
- Consideration of domain-specific properties of audio
- Consideration of issues specific to the FSDKaggle2019 dataset, including how to leverage the clean set and noisy set, potential domain mismatch, variable length of audio clips, etc.
- Computational efficiency
- Classification performance

The following strict rules apply for being eligible for the Judges' Award:

1.  Submit your system to DCASE Challenge (choose submission mode 2 or 3)
2.  Usage of both curated train set and noisy train set is mandatory, and novel ways of using the noisy train set will be especially valued
3.  Single model approaches are strongly preferred over ensembles. Occasionally, we could consider small ensembles of different models, if the approach is innovative.
4.  The Judges' Award Winner will be asked to publish the code as open-source
5.  Please report on the computational efficiency of your model by at least including the number of model parameters. Additional measures or information to satisfy the motivation of computational efficiency will be welcome.
6.  Please specify in the technical report the submission file (ie, *filename.csv*) corresponding to the candidate model to be considered for the Judges' Award.

Please check the [Submission Modes](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/submission-modes) page for instructions about how to get your submission considered for the Judges' Award. Note that the Judges' Award is a submission additional to the 2 scored private submissions eligible for the final private leaderboard. The Judges' Award winner will be announced after the competition ends and the organizers have reviewed all candidate submissions in this [discussion thread](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/88060). You can place your questions and comments about the Judges' Award in this same forum thread.

## Submission Modes

Here are the different submission modes that participants can decide to opt for:

- **Mode 1: Kaggle *only***: This is the most basic submission mode. For this mode submissions must happen using Kaggle Kernels. Check [Kernels Requirements](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/kernels-requirements) page for more information about how Kernel submissions work.

- **Mode 2: Kaggle + DCASE Challenge**: Because this competition is framed under the [DCASE Challenge](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/dcase), in addition to the normal Kaggle submission participants can choose to also submit their systems for consideration to [DCASE Challenge 2019 Task 2](http://dcase.community/challenge2019/task-audio-tagging). Submission to DCASE Challenge will require the writing of a technical report describing the system and filling a metadata file using specific templates that will be provided. Submissions to DCASE Challenge will happen using the DCASE submission system. Here are the [submission instructions for the DCASE Challenge](http://dcase.community/challenge2019/submission).

- **Mode 3: Kaggle + DCASE Challenge + DCASE Workshop**: Participating teams that want to get more involved with the academic community can also choose complement their submissions to Kaggle and the DCASE Challenge by writing a scientific paper to be sent to the [DCASE Workshop](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/dcase). Scientific papers sent to the DCASE Workshop are peer-reviewed and must be written following a specific template which will be provided. Note that submitting the scientific paper is *complementary* to the DCASE Challenge submission. To get your system considered in the DCASE Challenge you still need to follow the steps for submission in the challenge. Also note that the deadline for submitting the scientific paper to DCASE Workshop will be a couple of weeks after the deadline for the Kaggle/DCASE Challenge submissions. Here are the [submission instructions for the DCASE Challenge](http://dcase.community/challenge2019/submission). Submission instructions for the DCASE Workshop will be published in due time in the [DCASE Workshop website](http://dcase.community/workshop2019/).

**Note 1**: After the competition has ended and the results are analyzed, the DCASE Challenge website publishes a leaderboard with the results (see this example [leaderboard from last year's edition of this competition](http://dcase.community/challenge2018/task-general-purpose-audio-tagging-results)). Only participants that opted for submission modes 2 or 3 will appear in this leaderboard, hence the results might be different from those shown in Kaggle's leaderboard. The DCASE leaderboard is computed using the same evaluation metric and test set as in Kaggle's leaderboard.

**Note 2**: Except for the Judges' Award, other monetary prizes are given based on the Kaggle leaderboard. Therefore to opt for these prizes there is no need to submit systems to DCASE. Nevertheless, we strongly encourage all contestants to submit to DCASE (Challenge and/or Workshop) in order to share your work with a greater research community.

**Note 3**: To be considered for the Judges' Award participants must choose to submit their systems using submission modes 2 or 3. Additional specific criteria for being considered for the Judges' Award is given in the Judges' Award page.

Please, place any questions you might have about submission modes in this [discussion thread](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/88061).

## Kernels Requirements

![Kerneler](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Kernels-only competition

Submissions to this competition must be made through Kernels. You may train your model locally or in a Kernel, but you must submit your inference Kernel as your submission.\
The competition will take place in two stages. During the second stage, Kaggle will re-run your model on an unseen test set.\
In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- CPU Kernel <= 4 hours run-time
- GPU Kernel <= 1 hours run-time
- No internet access enabled
- External data and pre-trained models are not allowed. If you train your model offline, you *may* upload your model as a Kaggle dataset for use in creating your inference Kernel.
- No custom packages enabled in kernels
- Submission file must be named "submission.csv"

Please see the [Kernels-only FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ) for more information on how to submit.

## Citation

Addison Howard, Eduardo Fonseca, Frederic Font, inversion, Manoj Plakal. (2019). Freesound Audio Tagging 2019. Kaggle. https://kaggle.com/competitions/freesound-audio-tagging-2019


# Dataset Description

## Citation

If you use the FSDKaggle2019 dataset or baseline code, please cite our DCASE 2019 paper:

> Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Serra. *Audio tagging with noisy labels and minimal supervision*. In Proceedings of DCASE2019 Workshop, NYC, US (2019). URL: <https://arxiv.org/abs/1906.02975>

You can also consider citing our ISMIR 2017 paper, which describes how we gathered the manual annotations included in FSDKaggle2019.

> Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry\
> Bogdanov, Andrés Ferraro, Sergio Oramas, Alastair Porter, and Xavier\
> Serra. *Freesound datasets: a platform for the creation of open audio datasets*. In Proceedings of the 18th International Society for Music\
> Information Retrieval Conference (ISMIR 2017), pp 486-493. Suzhou,\
> China, 2017.

---

## Detected corrupted files in the curated train set

The following 5 audio files in the curated train set have a wrong label, due to a bug in the file renaming process:\
`f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav`

The audio file `1d44b0bd.wav` in the curated train set was found to be corrupted (contains no signal) due to an error in format conversion.

---

## Task Motivation

Current machine learning techniques require large and varied datasets in order to provide good performance and generalization. However, manually labelling a dataset is time-consuming, which limits its size. Websites like Freesound or Flickr host large volumes of user-contributed audio and metadata, and labels can be inferred automatically from the metadata and/or making predictions with pre-trained models. Nevertheless, these automatically inferred labels might include a substantial level of label noise.

The main research question addressed in this competition is how to adequately exploit a small amount of reliable, manually-labeled data, and a larger quantity of noisy web audio data in a multi-label audio tagging task with a large vocabulary setting. In addition, since the data comes from different sources, the task encourages domain adaptation approaches to deal with a potential domain mismatch.

## Audio Dataset

The dataset used in this challenge is called **FSDKaggle2019**, and it employs audio clips from the following sources:

- Freesound Dataset ([FSD](https://annotator.freesound.org/fsd/)): a dataset being collected at the [MTG-UPF](https://www.upf.edu/web/mtg) based on [Freesound](https://freesound.org/) content organized with the [AudioSet Ontology](https://research.google.com/audioset////////ontology/index.html)
- The soundtracks of a pool of Flickr videos taken from the [Yahoo Flickr Creative Commons 100M dataset (YFCC)](http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/)

The audio data is labeled using a vocabulary of 80 labels from Google's [AudioSet Ontology](https://research.google.com/audioset////////ontology/index.html) [1], covering diverse topics: Guitar and other Musical instruments, Percussion, Water, Digestive, Respiratory sounds, Human voice, Human locomotion, Hands, Human group actions, Insect, Domestic animals, Glass, Liquid, Motor vehicle (road), Mechanisms, Doors, and a variety of Domestic sounds. The full list of categories can be inspected in the `sample_submission.csv` at the the bottom of this page.

### Ground Truth Labels

The ground truth labels are provided at the clip-level, and express the presence of a sound category in the audio clip, hence can be considered *weak* labels or tags. Audio clips have variable lengths (roughly from 0.3 to 30s, see more details below).

The audio content from [FSD](https://annotator.freesound.org/fsd/) has been manually labeled by humans following a data labeling process using the [Freesound Annotator](https://annotator.freesound.org/) platform. Most labels have inter-annotator agreement but not all of them. More details about the data labeling process and the [Freesound Annotator](https://annotator.freesound.org/) can be found in [2].

The [YFCC](http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/) soundtracks were labeled using automated heuristics applied to the audio content and metadata of the original Flickr clips. Hence, a substantial amount of label noise can be expected. The label noise can vary widely in amount and type depending on the category, including in- and out-of-vocabulary noises. More information about some of the types of label noise that can be encountered is available in [3].

### Format and License

All clips are provided as uncompressed PCM 16 bit, 44.1 kHz, mono audio files. All clips used in this competition are released under Creative Commons (CC) licenses, some of them requiring attribution to their original authors and some forbidding further commercial reuse. In order to be able to comply with the CC licenses terms, a full list of audio clips with their associated licenses and a reference to the original content (in Freesound or Flickr) will be published at the end of the competition. Until then, the provided audio files can only be used for the sole purpose of participating in the competition.

## Train set

The **train set** is meant to be for system development. The idea is to limit the supervision provided (i.e., the manually-labeled data), thus promoting approaches to deal with label noise. The train set is composed of **two subsets** as follows:

### Curated subset

The curated subset is a small set of manually-labeled data from [FSD](https://annotator.freesound.org/fsd/).

- Number of clips/class: 75 except in a few cases (where there are less)
- Total number of clips: 4970
- Avge number of labels/clip: 1.2
- Total duration: 10.5 hours

The duration of the audio clips ranges from 0.3 to 30s due to the diversity of the sound categories and the preferences of Freesound users when recording/uploading sounds. It can happen that a few of these audio clips present additional acoustic material beyond the provided ground truth label(s).

### Noisy subset

The noisy subset is a larger set of noisy web audio data from Flickr videos taken from the [YFCC](http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/) dataset [5].

- Number of clips/class: 300
- Total number of clips: 19815
- Avge number of labels/clip: 1.2
- Total duration: ~80 hours

The duration of the audio clips ranges from 1s to 15s, with the vast majority lasting 15s.

Considering the numbers above, per-class data distribution available for training is, for most of the classes, 300 clips from the noisy subset and 75 clips from the curated subset, which means 80% noisy - 20% curated at the clip level (not at the audio duration level, considering the variable-length clips).

## Test set

The **test set** is used for system evaluation and consists of manually-labeled data from [FSD](https://annotator.freesound.org/fsd/). Since most of the train data come from [YFCC](http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/), some acoustic domain mismatch between the train and test set can be expected. All the acoustic material present in the test set is labeled, except human error, considering the vocabulary of 80 classes used in the competition.

The test set is split into two subsets, for the **public** and **private** leaderboards. In this competition, the submission is to be made through Kaggle Kernels. Only the test subset corresponding to the *public* leaderboard is provided (without ground truth).

Submissions must be made with inference models running in Kaggle Kernels. However, participants can decide to train also in the Kaggle Kernels or offline (see [Kernels Requirements](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/kernels-requirements) for details).

This is a kernels-only competition with two stages. The first stage comprehends the submission period until the deadline on June 10th. After the deadline, in the second stage, Kaggle will rerun your selected kernels on an unseen test set. **The second-stage test set is approximately three times the size of the first**. You should plan your kernel's memory, disk, and runtime footprint accordingly.

### Files

- **train_curated.csv** - ground truth labels for the curated subset of the training audio files (see Data Fields below)
- **train_noisy.csv** - ground truth labels for the noisy subset of the training audio files (see Data Fields below)
- **sample_submission.csv** - a sample submission file in the correct format, including the correct sorting of the sound categories; it contains the list of audio files found in the test.zip folder (corresponding to the public leaderboard)
- **train_curated.zip** - a folder containing the audio (.wav) training files of the curated subset
- **train_noisy.zip** - a folder containing the audio (.wav) training files of the noisy subset
- **test.zip** - a folder containing the audio (.wav) test files for the public leaderboard

### Columns

Each row of the train_curated.csv and train_noisy.csv files contains the following information:

- **fname**: the audio file name, eg, `0006ae4e.wav`
- **labels**: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.

References
==========

[1] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. "Audio set: An ontology and human-labeled dataset for audio events." In Proceedings of the International Conference on Acoustics, Speech and Signal Processing, 2017. [[PDF](https://ai.google/research/pubs/pub45857)]

[2] Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry Bogdanov, Andres Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra. "Freesound Datasets: A Platform for the Creation of Open Audio Datasets." In Proceedings of the International Conference on Music Information Retrieval, 2017. [[PDF](https://repositori.upf.edu/bitstream/handle/10230/33299/fonseca_ismir17_freesound.pdf)]

[3] Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, and Xavier Serra. "Learning Sound Event Classifiers from Web Audio with Noisy Labels." In Proceedings of the International Conference on Acoustics, Speech and Signal Processing, 2019. [[PDF](https://arxiv.org/abs/1901.01189)]

[4] Frederic Font, Gerard Roma, and Xavier Serra. "Freesound technical demo." Proceedings of the 21st ACM international conference on Multimedia, 2013. [https://freesound.org](https://freesound.org/)

[5] Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li, YFCC100M: The New Data in Multimedia Research, Commun. ACM, 59(2):64--73, January 2016