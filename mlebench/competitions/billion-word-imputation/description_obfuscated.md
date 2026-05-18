# Overview

Given sentences with one word removed, create a model capable of inserting back the correct missing word at the correct location in the sentence.

# Task

Levenshtein distance between submitted sentences and the original sentences.

# Submission Format

Your submission file should contain the sentence id and a predicted sentence. Use double quotes to escape the sentence text and two double quotes ("") for double quotes within a sentence.

The file should contain a header and have the following format:

```
id,"sentence"
1,"Former Dodgers manager , the team 's undisputed top ambassador , is going strong at 83 and serving up one great story after another ."
2,"8 parliamentary elections meant to restore democracy in this nuclear armed nation , a key ally against Islamic ."
3,"Sales of drink are growing 37 per cent month-on-month from a small base ."
etc...
```

# Dataset

You should use only the sentences in the training set to build you model.

We have removed one word from each sentence in the test set. The location of the removed word was chosen uniformly randomly and is never the first or last word of the sentence (in this dataset, the last word is always a period). You must attempt to submit the sentences in the test set with the correct missing word located in the correct location. 

## File descriptions

- **train.txt** - the training set, contains a large collection of English language sentences
- **test.txt** - the test set, contains a large number of sentences where one word has been removed
