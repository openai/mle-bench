# Overview

## Overview

The first [automated essay scoring competition](https://www.kaggle.com/competitions/asap-aes/overview) to tackle automated grading of student-written essays was twelve years ago. How far have we come from this initial competition? With an updated dataset and light years of new ideas we hope to see if we can get to the latest in automated grading to provide a real impact to overtaxed teachers who continue to have challenges with providing timely feedback, especially in underserved communities.

The goal of this competition is to train a model to score student essays. Your efforts are needed to reduce the high expense and time required to hand grade these essays. Reliable automated techniques could allow essays to be introduced in testing, a key indicator of student learning that is currently commonly avoided due to the challenges in grading.

## Description

Essay writing is an important method to evaluate student learning and performance. It is also time-consuming for educators to grade by hand. Automated Writing Evaluation (AWE) systems can score essays to supplement an educator’s other efforts. AWEs also allow students to receive regular and timely feedback on their writing. However, due to their costs, many advancements in the field are not widely available to students and educators. Open-source solutions to assess student writing are needed to reach every community with these important educational tools.

Previous efforts to develop open-source AWEs have been limited by small datasets that were not nationally diverse or focused on common essay formats. The first Automated Essay Scoring competition scored student-written short-answer responses, however, this is a writing task not often used in the classroom. To improve upon earlier efforts, a more expansive dataset that includes high-quality, realistic classroom writing samples was required. Further, to broaden the impact, the dataset should include samples across economic and location populations to mitigate the potential of algorithmic bias.

In this competition, you will work with the largest open-access writing dataset aligned to current standards for student-appropriate assessments. Can you help produce an open-source essay scoring algorithm that improves upon the original [Automated Student Assessment Prize (ASAP)](https://www.kaggle.com/competitions/asap-aes/overview) competition hosted in 2012?

Competition host Vanderbilt University is a private research university in Nashville, Tennessee. For this competition, Vanderbilt has partnered with The Learning Agency Lab, an Arizona-based independent nonprofit focused on developing the science of learning-based tools and programs for the social good.

To ensure the results of this competition are widely available, winning solutions will be released as open source. More robust and accessible AWE options will help more students get the frequent feedback they need and provide educators with additional support, especially in underserved districts.

### Acknowledgments

Vanderbilt University and the Learning Agency Lab would like to thank the Bill & Melinda Gates Foundation, Schmidt Futures, and Chan Zuckerberg Initiative for their support in making this work possible.

![BMGF_logo](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/BMGF_logo_black_300dpi%20(1).jpg)

![Schmidt Futures Logo](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/Schmidt%20Futures%20Logo.png)

![CZI Logo](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/1200px-Chan_Zuckerberg_Initiative.svg.png)

## Evaluation

Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0 .

The quadratic weighted kappa is calculated as follows. First, an $N \times N$ histogram matrix $O$ is constructed, such that $O_{i, j}$ corresponds to the number of essay_id s $i$ (actual) that received a predicted value $j$. An $N-b y-N$ matrix of weights, $w$, is calculated based on the difference between actual and predicted values:

$$
w_{i, j}=\frac{(i-j)^2}{(N-1)^2}
$$

An $N$-by- $N$ histogram matrix of expected outcomes, $E$, is calculated assuming that there is no correlation between values.

This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that $E$ and $O$ have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as:

$$
\kappa=1-\frac{\sum_{i, j} w_{i, j} O_{i, j}}{\sum_{i, j} w_{i, j} E_{i, j}}
$$

### Submission File

For each `essay_id` in the test set, you must predict the corresponding `score` (described on the [Data](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data) page). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Timeline

- **April 2, 2024** - Start Date.
- **June 25, 2024** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **June 25, 2024** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **July 2 2024** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

### Leaderboard Prizes

- 1st Place - $ 12,000
- 2nd Place - $ 8,000
- 3rd Place - $ 5,000

### Efficiency Prizes

- 1st Place - $ 12,000
- 2nd Place - $ 8,000
- 3rd Place - $ 5,000

Please see [**Efficiency Prize Evaluation**](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview/efficiency-prize-evaluation) for details on how the Efficiency Prize will be awarded. Winning a Leaderboard Prize does not preclude you from winning an Efficiency Prize.

## Code Requirements

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Efficiency Prize Evaluation

### Efficiency Prize

We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy. Such models have a stronger carbon footprint and frequently prove difficult to utilize in real-world educational contexts. We hope to use these models to help educational organizations, which have limited computational capabilities.

For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance.

To be eligible for an Efficiency Prize, a submission:

- Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the [**My Submissions**](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/submissions) tab.
- Must be ranked on the Private Leaderboard higher than the `sample_submission.csv` benchmark.
- Must not have a GPU enabled. **The Efficiency Prize is CPU Only.**

All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize.

An Efficiency Prize will be awarded to eligible submissions according to how they are ranked by the following evaluation metric on the private test data. See the [**Prizes**](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview/prizes) tab for the prize awarded to each rank. More details may be posted via discussion forum updates.

### Evaluation Metric

We compute a submission's **efficiency score** by:

$$
\text { Efficiency }=\frac{\text { Kappa }}{\text { Base }- \text { max Kappa }}+\frac{\text { RuntimeSeconds }}{32400}
$$

where Kappa is the submission's score on the [main competition metric](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview/evaluation), BaseBase is the score of the "baseline" submission of the mean of each target on the training set, minKappaminKappa is the minimum Kappa of all submissions on the Private Leaderboard, and RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score.

During the training period of the competition, you may see a leaderboard for the *public* test data in the following notebook, updated daily: [**Efficiency Leaderboard**](https://www.kaggle.com/code/inversion/automated-essay-scoring-2-0-efficiency-lb). After the competition ends, we will update this leaderboard with efficiency scores on the *private* data. During the training period, this leaderboard will show only the rank of each team, but not the complete score.

## Citation

Scott Crossley, Perpetual Baffour, Jules King, Lauryn Burleigh, Walter Reade, Maggie Demkin. (2024). Learning Agency Lab - Automated Essay Scoring 2.0. Kaggle. https://kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2

# Data

## Dataset Description

The competition dataset comprises about 24000 student-written argumentative essays. Each essay was scored on a scale of 1 to 6 ([Link to the Holistic Scoring Rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf)). Your goal is to predict the score an essay received from its text.

### File and Field Information

- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`. (**Note**: The rerun test set has approximately 8k observations.)
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

Please note that this is a [**Code Competition**](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview/code-requirements).