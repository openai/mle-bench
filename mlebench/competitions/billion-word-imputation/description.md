# Overview

## Overview

### Description

![corpus](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3927/media/corpus.png)This competition uses the billion-word benchmark corpus provided by [Chelba et al.](http://arxiv.org/abs/1312.3005) for language modeling. Rather than ask participants to create a classic language model and evaluate sentence probabilities -- a task which is difficult to faithfully score in Kaggle's supervised ML setting -- we have introduced a variation on the language modeling task.

For each sentence in the test set, we have removed exactly one word. Participants must create a model capable of inserting back the correct missing word at the correct location in the sentence. Submissions are scored using an edit distance to allow for partial credit.

We extend our thanks to authors who created this corpus and shared it for the research community to use. Please cite this paper if you use this dataset in your research: *Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn: One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling, CoRR, 2013.*

Note: the train/test split used in this competition is different than the published version used for language modeling. If you are creating full language models and scoring perplexity, you should download the official version of the corpus from the authors' website.

### Evaluation

Submissions are evaluated on the mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) between the sentences you submit and the original sentences in the test set.

*Note: due to the size and computations necessary to score submissions for this competition, scoring may take 5-10 minutes, and possibly longer if there are other submissions in front of yours. Please be patient!*

#### Submission File

Your submission file should contain the sentence id and a predicted sentence. To prevent parsing issues, you should use double quotes to escape the sentence text and two double quotes ("") for double quotes within a sentence. Note that test.csv is a valid submission file itself.

The file should contain a header and have the following format:

```
id,"sentence"
1,"Former Dodgers manager , the team 's undisputed top ambassador , is going strong at 83 and serving up one great story after another ."
2,"8 parliamentary elections meant to restore democracy in this nuclear armed nation , a key ally against Islamic ."
3,"Sales of drink are growing 37 per cent month-on-month from a small base ."
etc...
```

### Citation

Will Cukierski. (2014). Billion Word Imputation. Kaggle. https://kaggle.com/competitions/billion-word-imputation

# Data

## Dataset Description

The data for this competition is a large corpus of English language sentences. You should use only the sentences in the training set to build you model.

We have removed one word from each sentence in the test set. The location of the removed word was chosen uniformly randomly and is never the first or last word of the sentence (in this dataset, the last word is always a period). You must attempt to submit the sentences in the test set with the correct missing word located in the correct location. 

Note: the train/test split used in this competition is different than the published version used for language modeling. If you are creating full language models and scoring perplexity, you should download the official version of the corpus from the authors' website.

## File descriptions

- **train.txt** - the training set, contains a large collection of English language sentences
- **test.txt** - the test set, contains a large number of sentences where one word has been removed
