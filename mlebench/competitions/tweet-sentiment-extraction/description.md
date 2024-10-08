# Overview

## Description

*"My ridiculous dog is amazing."* [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

Help build your skills in this important area with this broad dataset of tweets. Work on your technique to grab a top spot in this competition. What words in tweets support a positive, negative, or neutral sentiment? How can you help make that determination using machine learning tools?

In this competition we've extracted support phrases from [Figure Eight's Data for Everyone platform](https://www.figure-eight.com/data-for-everyone/). The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international licence. Your objective in this competition is to construct a model that can do the same - look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

## Evaluation

The metric in this competition is the [word-level Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index). A good description of Jaccard similarity for strings is [here](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50). A Python implementation based on the links above, and matched with the output of the C# implementation on the back end, is provided below.

```python
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric, then, is:

score $=\frac{1}{n} \sum_{i=1}^n \text{jaccard} \left(g t \_i, d t \_i\right)$

where:

- $n$ = number of documents

- $\text{jaccard}$ = the function provided above

- $gt\_i$ = the ith ground truth

- $dt\_i$= the ith prediction

## Submission File

For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

### Timeline

- June 9, 2020 - Entry deadline. You must accept the competition rules before this date in order to compete.

- June 9, 2020 - Team Merger deadline. This is the last day participants may join or merge teams.

- June 16, 2020 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

- 1st Place - $7,000
- 2nd Place - $5,000
- 3rd Place - $3,000

## Code Requirements

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- CPU Kernel <= 3 hours run-time
- GPU Kernel <= 3 hours run-time
- No internet access enabled
- External data, freely & publicly available, is allowed, including pre-trained models
- Submission file must be named "submission.csv"\
    Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ) for more information on how to submit.

### Citation

Maggie, Phil Culliton, Wei Chen. (2020). Tweet Sentiment Extraction. Kaggle. https://kaggle.com/competitions/tweet-sentiment-extraction


# Data

## Dataset Description

### What files do I need?

You'll need **train.csv**, **test.csv**, and **sample_submission.csv**.

### What should I expect the data format to be?

Each row contains the `text` of a tweet and a `sentiment` label. In the training set you are provided with a word or phrase drawn from the tweet (`selected_text`) that encapsulates the provided sentiment.

Make sure, when parsing the CSV, to remove the beginning / ending quotes from the `text` field, to ensure that you don't include them in your training.

### What am I predicting?

You're attempting to predict the word or phrase from the tweet that exemplifies the provided sentiment. The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.). The format is as follows:

`<id>,"<word or phrase that supports the sentiment>"`

For example:

```
2,"very good"
5,"I am neutral about this"
6,"bad"
8,"if you say so!"
etc.
```

Files
-----

- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment