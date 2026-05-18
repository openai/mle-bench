# Overview

## Description

Looking for a data science position at Facebook?  After two successful prior Kaggle competitions, Facebook continues their mission to identify the best data scientists and software engineers that Kaggle has to offer. In this third installment, they seek candidates who have experience text mining large amounts of data.

![tags.jpg](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3539/media/tags.jpg)

This competition tests your text skills on a large dataset from the Stack Exchange sites.  The task is to predict the tags (a.k.a. keywords, topics, summaries), given only the question text and its title. The dataset contains content from disparate stack exchange sites, containing a mix of both technical and non-technical questions.

*Positions are available in Menlo Park, Seattle, New York City, and London; candidates must have, or be eligible to obtain, authorization to work in the US or UK.*

**Please note: you must compete as an individual in recruiting competitions. You may only use the data provided to make your predictions. Crawling stack exchange sites to look up answers is not permitted. Facebook will review the code of the top participants before deciding whether to offer an interview.**

**This competition counts towards rankings & achievements.**  If you wish to be considered for an interview at Facebook, check the box "Allow host to contact me" when you make your first entry.

### Acknowledgments

We thank Stack Exchange (and its users) for generously releasing the source dataset through its [Creative Commons Data Dumps](http://blog.stackoverflow.com/category/cc-wiki-dump). All data is licensed under the [cc-by-sa](http://creativecommons.org/licenses/by-sa/2.5/) license.

## Prizes

Highly ranked contestants who indicate their interest will be considered by Facebook for interviews, based on their work in the competition and additional background. Positions are available in Menlo Park, Seattle, New York City, and London. Candidates must have, or be eligible to obtain, authorization to work in the US or UK.

## Evaluation

The evaluation metric for this competition is [Mean F1-Score](https://www.kaggle.com/wiki/MeanFScore). The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision p and recall r. Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn). The F1 score is given by:

$F_1 = 2 \frac{p * r}{p + r}$ where $p = \frac{tp}{tp + fp}$ and $r = \frac{tp}{tp + fn}$

The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

To receive credit, the tag you predict must be an exact match, regardless of whether you believe the tags are synonyms. Why aren't synonyms counted?

1. Giving out a list of candidate synonyms is a potential source of [leakage](https://www.kaggle.com/wiki/Leakage)
2. Synonyms are subjective, and there are "subjectively many" synonyms for a given tag
3. Participants are equally penalized for predicting a synonym of a correct tag, so the task can be framed as not only predicting a tag, but also modeling the distribution(s) of potential synonyms

### Submission Instructions

For every question in the test set, your submission file should contain two columns: Id and Tags. Id is the unique identifier of a question in test.csv. Tags should be a space-delimited list predicted tags. You should maintain the order of questions.

The file should contain a header and have the following format:

```
Id,Tags
1,"c++ javaScript"
2,"php python mysql"
3,"django"
etc.
```

## Citation

Alex Hollander, Anirudh Ranganath, Itamar Rosenn, Jim Dullaghan, John P. Costella, Junfeng Pan, Will Cukierski. (2013). Facebook Recruiting III - Keyword Extraction. Kaggle. https://kaggle.com/competitions/facebook-recruiting-iii-keyword-extraction

# Dataset Description

All of the data is in 2 files: Train and Test.

**Train.csv** contains 4 columns: Id,Title,Body,Tags

- Id - Unique identifier for each question
- Title - The question's title
- Body - The body of the question
- Tags - The tags associated with the question (all lowercase, should not contain tabs '\t' or ampersands '&')

**Test.csv** contains the same columns but without the Tags, which you are to predict.

The questions are randomized and contains a mix of verbose text sites as well as sites related to math and programming. The number of questions from each site may vary, and no filtering has been performed on the questions (such as closed questions).

## Acknowledgements

We thank Stack Exchange (and its users) for generously releasing the source dataset through its [Creative Commons Data Dumps](http://blog.stackoverflow.com/category/cc-wiki-dump). All data is licensed under the [cc-by-sa](http://creativecommons.org/licenses/by-sa/2.5/) license.