# Overview

## Description

### Goal of the Competition

Join the $1,000,000+ [Vesuvius Challenge](https://scrollprize.org/) to resurrect an ancient library from the ashes of a volcano. In this competition you are tasked with detecting ink from 3D X-ray scans and reading the contents. Thousands of scrolls were part of a library located in a Roman villa in Herculaneum, a town next to Pompeii. This villa was buried by the Vesuvius eruption nearly 2000 years ago. Due to the heat of the volcano, the scrolls were carbonized, and are now impossible to open without breaking them. These scrolls were discovered a few hundred years ago and have been waiting to be read using modern techniques. There is a $700,000 grand prize available to the first team that can read these scrolls from a 3D X-ray scan. This Kaggle competition hosts the **Ink Detection Progress Prize**.

Prizes breakdown:

- [Grand Prize](https://scrollprize.org/) - $700,000
- Ink Detection Progress Prize on Kaggle - $100,000 in prizes
- [Segmentation Tooling Prize](https://scrollprize.org/segmentation) - $45,000 in prizes
- [First Letters Prize](https://scrollprize.org/first_letters) - $50,000 in prizes
- To be announced - $200,000+

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F13132488%2Fa620bbf1c264b06aa75ae59195264df0%2Fmaxresdefault.webp?generation=1676936353844811&alt=media)

*One of the scrolls, which cannot be physically opened [(source)](https://www.youtube.com/watch?v=PpNq2cFotyY)*

### This Kaggle competition

This Kaggle competition hosts the **Ink Detection Progress Prize** ($100,000 in prizes), which is about the sub-problem of detecting ink from 3d x-ray scans of fragments of papyrus which became detached from some of the excavated scrolls. This subcontest is run on Kaggle since it's a more traditional data science / machine learning problem of building a model that can be verified against known ground truth data.

The ink used in the Herculaneum scrolls does not show up readily in X-ray scans. But we have found that machine learning models can detect it. Luckily, we have ground truth data. Since the discovery of the Herculaneum Papyri almost 300 years ago, people have tried opening them, often with disastrous results. Many scrolls were destroyed in this process, but ink can be seen on some broken-off fragments, especially under infrared light.

The dataset contains 3d x-ray scans of four such fragments at 4µm resolution, made using a particle accelerator, as well as infrared photographs of the surface of the fragments showing visible ink. These photographs have been aligned with the x-ray scans. We also provide hand-labeled binary masks indicating the presence of ink in the photographs.

Get started exploring this problem:

- [Tutorial](https://scrollprize.org/tutorial4)
- [Notebook](https://www.kaggle.com/jpposma/vesuvius-challenge-ink-detection-tutorial/edit)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F13132488%2Fa80987e5e9973d4f338e18fb8583abee%2FScreen%20Shot%202023-02-20%20at%2015.51.03.png?generation=1676937111036490&alt=media)

### About the competition

Vesuvius Challenge is created by [Nat Friedman](https://nat.org/) and [Daniel Gross](https://dcgross.com/), and is hosted in collaboration with [EduceLab](https://educelab.engr.uky.edu/). We appreciate Kaggle's help with this Ink Detection competition. For more information on the organizing team, partner organizations, and other background, check out [scrollprize.org](https://scrollprize.org/).

> **This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/vesuvius-challenge-ink-detection/overview/code-requirements) for details.**

## Evaluation

We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). We also show how to output a file of this format in the [Ink Detection tutorial](https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial).

## Timeline

- **March 15, 2023** - Start Date.
- **June 7, 2023** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **June 7, 2023** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **June 14, 2023** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

[**Vesuvius Challenge**](https://www.scrollprize.org)

$700,000 for the complete effort to "unwrap" the ancient scrolls, open source prizes, and more in prizes TBA.

For this Ink Detection progress prize on Kaggle, we're awarding $100,000 for the top 10 winners on the leaderboard:

- 1st place: $25,000
- 2nd place: $20,000
- 3rd place: $15,000
- 4th place: $10,000
- 5th--10th: $5,000

For more details on the other prizes, check out [scrollprize.org](https://www.scrollprize.org).

## Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, except photos or other data associated with the fragments
- All your work should be fully reproducible by the contest organizers
- Offline pretraining (on external computers) is allowed. Just make sure to document your process so we can reproduce it if you win.
- Submission file must be named `submission.csv`
- For the full rules, see the [rules page](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/rules)

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Citation

AlexLourenco, Brent Seales, Christy Chapman, Daniel Havir, Ian Janicki, JP Posma, Nat Friedman, Ryan Holbrook, Seth P., Stephen Parsons, Will Cukierski. (2023). Vesuvius Challenge - Ink Detection. Kaggle. https://kaggle.com/competitions/vesuvius-challenge-ink-detection

# Data

## Dataset Description

Your challenge is to recover where ink is present from 3d x-ray scans of detached fragments of ancient papyrus scrolls. This is an important subproblem in the overall task of solving the [Vesuvius Challenge](https://scrollprize.org).

This is a [Code Competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview/code-requirements). When your submitted notebook is scored, the actual test data will be made available to your notebook. Before that, the test/ directory will contain dummy data. This is done to keep the actual test data secret.

### Files

- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should [make your submission in](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview/evaluation).
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**. See the [evaluation page](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview/evaluation) for information.

For an example program, see the [tutorial notebook](https://www.kaggle.com/jpposma/vesuvius-challenge-ink-detection-tutorial/).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F13132488%2Fe6a701bf90319edd1632b97b27266f39%2FScreen%20Cast%202023-02-09%20at%2013.59.58.gif?generation=1676075979829484&alt=media)
