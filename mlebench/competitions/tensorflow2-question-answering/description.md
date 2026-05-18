# Overview

## Description

*"Why is the sky blue?"*

This is a question an [open-domain question answering](https://en.wikipedia.org/wiki/Question_answering#Open_domain_question_answering) (QA) system should be able to respond to. QA systems emulate how people look for information by reading the web to return answers to common questions. Machine learning can be used to improve the accuracy of these answers.

Existing natural language models have been focused on extracting answers from a short paragraph rather than reading an entire page of content for proper context. As a result, the responses can be complicated or lengthy. A good answer will be both succinct and relevant.

In this competition, your goal is to predict short and long answer responses to real questions about Wikipedia articles. The dataset is provided by [Google's Natural Questions](https://ai.google.com/research/NaturalQuestions/dataset), but contains its own unique private test set. A [visualization of examples](https://ai.google.com/research/NaturalQuestions/visualization) shows long and---where available---short answers. In addition to prizes for the top teams, there is a special set of awards for using TensorFlow 2.0 APIs.

If successful, this challenge will help spur the development of more effective and robust QA systems.

## About TensorFlow

TensorFlow is an open source platform for machine learning. With TensorFlow 2.0, tf.keras is the preferred high-level API for TensorFlow, to make model building easier and more intuitive. You may use the [tf.keras built-in compile()/fit() methods](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_i_using_build-in_training_evaluation_loops), or write your own [custom training loops](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch). See the [Effective TensorFlow 2.0 guide](https://www.tensorflow.org/guide/effective_tf2) and the [tf.keras guide](https://www.tensorflow.org/guide/keras/) for more details.

TensorFlow 2.0 was [recently released](https://medium.com/tensorflow/tensorflow-2-0-is-now-available-57d706c2a9ab) and this competition is to challenge Kagglers to use TensorFlow 2.0's APIs focused on usability, and easier, more intuitive development, to make advancements on Question Answering.

## Evaluation

Submissions are evaluated using [micro F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) between the predicted and expected answers. Predicted long and short answers must match exactly the token indices of one of the ground truth labels ((or match YES/NO if the question has a yes/no short answer). There may be up to five labels for long answers, and more for short. If no answer applies, leave the prediction blank/null.

The metric in this competition diverges from the [original metric](https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py) in two key respects: 1) short and long answer formats do not receive separate scores, but are instead combined into a [micro F1 ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)score across both formats, and 2) this competition's metric does *not* use `confidence` scores to find an optimal threshold for predictions.

Additional detail:

$F_1=2 * \frac{\text { precision } * \text { recall }}{\text { precision }+ \text { recall }}$

where:

$\begin{gathered}\text { precision }=\frac{T P}{T P+F P} \\ \text { recall }=\frac{T P}{T P+F N}\end{gathered}$

and:

> TP = the predicted indices match one of the possible ground truth indices\
> FP = the predicted indices do NOT match one of the possible ground truth indices, OR a prediction has been made where no ground truth exists\
> FN = no prediction has been made where a ground truth exists

In "micro" F1, both long and short answers count toward an overall precision and recall, which is then used to calculate a single F1 value. (In contrast, in "macro" F1 a separate F1 score is calculated for each class / label and then averaged.)

## Submission File

For each ID in the test set, you must predict a) a set of start:end token indices, b) a YES/NO answer if applicable (short answers ONLY), or c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:

```
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
-6998273848279890840_long,
-6998273848279890840_short,NO
```

## Prizes

link

keyboard_arrow_up

The following prizes will be awarded on the basis of private leaderboard rank.

Leaderboard Prizes:

- First Prize: $12,000

- Second Prize: $8,000

- Third Prize: $5,000

* * * * *

This competition is sponsored by the TensorFlow team. TensorFlow 2.0 was [recently released](https://medium.com/tensorflow/tensorflow-2-0-is-now-available-57d706c2a9ab) and this competition is to challenge Kagglers to use TensorFlow 2.0's APIs focused on usability, and easier, more intuitive development, to make advancements on the Natural Questions Document Understanding problem. Show off what you can do with TensorFlow 2.0 and you could win the following prizes.

TensorFlow 2.0 Prizes:

- First Prize: $12,000

- Second Prize: $8,000

- Third Prize: $5,000

To be eligible for the TensorFlow 2.0 prizes, your code must use TensorFlow 2.0. More specifically:

- it should run with a public release of TensorFlow 2.0 installed, and

- should not use any tf.compat.v1 module symbols and should not raise deprecation warnings

At the end of the competition, we'll publish a form for participants who have met these eligibility requirements to request consideration for these special set of TF2.0 prizes.

## Timeline

- January 15, 2020 - Team Merger deadline. This is the last day participants may join or merge teams.
- January 15, 2020 - Entry deadline. You must accept the competition rules before this date in order to compete.
- January 22, 2020 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Getting Started

Here are some resources to help you get started:

- [Starter notebook](https://www.kaggle.com/philculliton/using-tensorflow-2-0-w-bert-on-nq) using BERT on the dataset. *Note: This baseline uses code that was migrated from TF1.x. Be aware that it contains use of tf.compat.v1, which is not permitted to be eligible for [TF2.0 prizes in this competition](https://www.kaggle.com/c/tensorflow2-question-answering/overview/prizes). It is intended to be used as a starting point, but we're excited to see how much better you can using TF2.0!*
- Some additional TensorFlow 2.0 resources:
    - [Effective TensorFlow 2.0 Guide](https://www.tensorflow.org/guide/effective_tf2)
    - [Migrating code from TF1.x to TF2.0](https://www.tensorflow.org/guide/migrate) as well as an [upgrade script](https://www.tensorflow.org/guide/upgrade)
    - [tf.keras Guide](https://www.tensorflow.org/guide/keras/)
    - [Kaggle's announcement about TF2.0 released in notebooks](https://www.kaggle.com/general/114059)
- Information on the [TF2.0 prizes](https://www.kaggle.com/c/tensorflow2-question-answering/overview/prizes) you an earn in this competition
- The [Natural Questions](https://ai.google.com/research/NaturalQuestions) site for more information about their dataset or to download directly from them. Note that this competition's private test set is completely different from any of the datasets used by Natural Questions.
- Google Cloud Platform (GCP) Credits: For entrants in this competition, we are distributing $300 Coupons for GCP Credit. [Requests can be made by submitting this form](https://www.kaggle.com/GCP-Credits-Tensorflow2-QA) by November 18, 2019. We'll distribute the coupons by November 25th.
    - You must have made a submission to this competition, or your request will be rejected.
    - Only one person per team will be issued a Coupon. This will be verified and multiple requests from the same team will be de-duplicated.

## Code Requirements

![Kerneler](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a code competition

Submissions to this competition must be made through Kaggle Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- GPU Notebooks <= 3 hours run-time
- CPU Notebooks <= 9 hours run-time
- Internet access off
- External data is allowed (including training offline and uploading the model)
- No custom packages enabled in kernels
- Submission files must be named "submission.csv"

Please see the [Code competition FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ) for more information on how to submit.

## Citation

keyboard_arrow_up

fchollet, Julia Elliott, JWhyW24, Paige Bailey, Phil Culliton, raguptamd, TomK. (2019). TensorFlow 2.0 Question Answering. Kaggle. https://kaggle.com/competitions/tensorflow2-question-answering


# Data

## Dataset Description

In this competition, we are tasked with selecting the best short and long answers from Wikipedia articles to the given questions.

### What should I expect the data format to be?

Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.

### What am I predicting?

For each article + question pair, you must predict / select long and short form answers to the question drawn *directly from the article*.

- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.

There is more detail about the data and what you're predicting [on the Github page for the Natural Questions dataset](https://github.com/google-research-datasets/natural-questions/blob/master/README.md). This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.

### File descriptions

- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
- **simplified-nq-kaggle-test.jsonl** - the test data, in newline-delimited JSON format.
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields

- **document_text** - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
- **question_text** - the question to be answered
- **long_answer_candidates** - a JSON array containing all of the plausible long answers.
- **annotations** - a JSON array containing all of the correct long + short answers. Only provided for train.
- **document_url** - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
- **example_id** - unique ID for the sample.