# Overview

## Description

Can you extract meaning from a large, text-based dataset derived from inventions? Here's your chance to do so.

The U.S. Patent and Trademark Office (USPTO) offers one of the largest repositories of scientific, technical, and commercial information in the world through its [Open Data Portal](https://developer.uspto.gov/about-open-data). Patents are a form of [intellectual property](https://www.uspto.gov/patents/basics/general-information-patents) granted in exchange for the public disclosure of new and useful inventions. Because patents undergo an intensive [vetting process](https://www.uspto.gov/sites/default/files/documents/InventionCon2020_Understanding_the_Patent_Examination_Process.pdf) prior to grant, and because the history of U.S. innovation spans over two centuries and 11 million patents, the U.S. patent archives stand as a rare combination of data volume, quality, and diversity.

> "The USPTO serves an American innovation machine that never sleeps by granting patents, registering trademarks, and promoting intellectual property around the globe. The USPTO shares over 200 years' worth of human ingenuity with the world, from lightbulbs to quantum computers. Combined with creativity from the data science community, USPTO datasets carry unbounded potential to empower AI and ML models that will benefit the progress of science and society at large."
>
> --- USPTO Chief Information Officer Jamie Holcombe

In this competition, you will train your models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents. Determining the semantic similarity between phrases is critically important during the patent search and examination process to determine if an invention has been described before. For example, if one invention claims "television set" and a prior publication describes "TV set", a model would ideally recognize these are the same and assist a patent attorney or examiner in retrieving relevant documents. This extends beyond paraphrase identification; if one invention claims a "strong material" and another uses "steel", that may also be a match. What counts as a "strong material" varies per domain (it may be steel in one domain and ripstop fabric in another, but you wouldn't want your parachute made of steel). We have included the Cooperative Patent Classification as the technical domain context as an additional feature to help you disambiguate these situations.

Can you build a model to match phrases in order to extract contextual information, thereby helping the patent community connect the dots between millions of patent documents?

> This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching/overview/code-requirements) for details.

## Evaluation

Submissions are evaluated on the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between the predicted and actual similarity `score`s.

## Submission File

For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Timeline

- March 21, 2022 - Start Date.
- June 13, 2022 - Entry Deadline. You must accept the competition rules before this date in order to compete.
- June 13, 2022 - Team Merger Deadline. This is the last day participants may join or merge teams.
- June 20, 2022 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - $ 12,000
- 2nd Place - $ 8,000
- 3rd Place - $ 5,000

## Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook ≤ 9 hours run-time
- GPU Notebook ≤ 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## USPTO Open Data

## Acknowledgments

![USPTO logo](https://www.uspto.gov/profiles/uspto_gov/themes/uspto/images/USPTO-logo-RGB-stacked-1200px.png)

Data for this competition is derived from the public archives of the U.S. Patent and Trademark Office (USPTO). These archives, offered in machine-readable formats, have already enabled numerous AI research projects such as the [training of large-scale language models](https://arxiv.org/abs/2101.00027). Beyond enabling AI research, patent documents can also be used to understand the [dynamics of AI innovation at large](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset).

USPTO data is offered both in bulk and via API through the [USPTO Open Data Portal](https://developer.uspto.gov/). Additionally, a prototype API covering a subset of USPTO data is available at [PatentsView](https://patentsview.org/apis/purpose).

## Citation

Don Cenkci, Grigor Aslanyan, Ian Wetherbee, jm, Kiran Gunda, Maggie, Scott Beliveau, Will Cukierski. (2022). U.S. Patent Phrase to Phrase Matching . Kaggle. https://kaggle.com/competitions/us-patent-phrase-to-phrase-matching


# Data

## Dataset Description

In this dataset, you are presented pairs of phrases (an `anchor` and a `target` phrase) and asked to rate how similar they are on a scale from 0 (not at all similar) to 1 (identical in meaning). This challenge differs from a standard semantic similarity task in that similarity has been scored here within a patent's `context`, specifically its [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject to which the patent relates. For example, while the phrases "bird" and "Cape Cod" may have low semantic similarity in normal language, the likeness of their meaning is much closer if considered in the context of "house".

This is a code competition, in which you will submit code that will be run against an unseen test set. The unseen test set contains approximately 12k pairs of phrases. A small public test set has been provided for testing purposes, but is not used in scoring.

Information on the meaning of CPC codes may be found on the [USPTO website](https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html). The CPC version 2021.05 can be found on the [CPC archive website](https://www.cooperativepatentclassification.org/Archive).

## Score meanings

The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

> "Google Patent Phrase Similarity Dataset" by Google is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0)