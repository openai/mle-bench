# Overview

## Description

The goal of this competition is to understand the relationship between code and comments in Python notebooks. You are challenged to reconstruct the order of markdown cells in a given notebook based on the order of the code cells, demonstrating comprehension of which natural language references which code.

### Context

Research teams across Google and Alphabet are exploring new ways that machine learning can assist software developers, and want to rally more members of the developer community to help explore this area too. Python notebooks provide a unique learning opportunity, because unlike a lot of standard source code, notebooks often follow narrative format, with comment cells implemented in markdown that explain a programmer's intentions for corresponding code cells. An understanding of the relationships between code and markdown could lend to fresh improvements across many aspects of AI-assisted development, such as the construction of better data filtering and preprocessing pipelines for model training, or automatic assessments of a notebook's readability.

We have assembled a dataset of approximately 160,000 public Python notebooks from Kaggle and have teamed up with [X, the moonshot factory](https://x.company/) to design a competition that challenges participants to use this dataset of published notebooks to build creative techniques aimed at better understanding the relationship between comment cells and code cells.

![Image of Notebook Cells](https://storage.googleapis.com/kaggle-media/Images/notebook_cell_examples.png)

After the submission deadline, Kaggle and X will evaluate the performance of submitted techniques on new, previously unseen notebooks. We're excited to see how the insights learned from this competition affect the future of notebook authorship.

## Evaluation

Predictions are evaluated by the [Kendall tau correlation](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) between predicted cell orders and ground truth cell orders accumulated across the entire collection of test set notebooks.

Let $S$ be the number of swaps of adjacent entries needed to sort the predicted cell order into the ground truth cell order. In the worst case, a predicted order for a notebook with $n$ cells will need $\frac{1}{2} n(n-1)$ swaps to sort.

We sum the number of swaps from your predicted cell order across the entire collection of test set notebooks, and similarly with the worst-case number of swaps. We then compute the Kendall tau correlation as:

$K=1-4 \frac{\sum_i S_i}{\sum_i n_i\left(n_i-1\right)}$

You may find a Python implementation in this notebook: [Competition Metric - Kendall Tau Correlation](https://www.kaggle.com/ryanholbrook/competition-metric-kendall-tau-correlation).

## Submission File

For each `id` in the test set (representing a notebook), you must predict `cell_order`, the correct ordering of its cells in terms of the cell ids. The file should contain a header and have the following format:

```
id,cell_order
0009d135ece78d,ddfd239c c6cd22db 1372ae9b ...
0010483c12ba9b,54c7cab3 fe66203e 7844d5f8 ...
0010a919d60e4f,aafc3d23 80e077ec b190ebb4 ...
0028856e09c5b7,012c9d02 d22526d1 3ae7ece3 ...
etc.
```

## Timeline

**This is a two-stage competition, with a training stage and a collection stage, during which models will be rerun against notebooks which are yet to be authored.**

### Training Timeline

- May 11, 2022 - Start Date
- August 4, 2022 - Entry deadline. You must accept the competition rules before this date in order to compete.
- August 4, 2022 - Team Merger deadline. This is the last day participants may join or merge teams.
- August 11, 2022 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Collection Timeline:

Starting after the final submission deadline, there will be periodic updates to the leaderboard to reflect models' performance against newly authored notebooks on Kaggle.

- November 10, 2022 - Competition End Date - Winner's announcement

## Prizes

- 1st Place - $50,000
- 2nd Place - $40,000
- 3rd Place - $30,000
- 4th Place - $20,000
- 5th Place - $10,000

## Code Requirements

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Citation

Addison Howard, Alex Polozov, Bin Ni, Christopher Tirrell, MicHCR, Michele (pirroh) Catasta, Olivia Hatalsky, Rishabh Singh, Ryan Holbrook, Will Cukierski. (2022). Google AI4Code -- Understand Code in Python Notebooks. Kaggle. https://kaggle.com/competitions/AI4Code


# Dataset Description

The dataset for this competition comprises about 160,000 Jupyter notebooks published by the Kaggle community. Jupyter notebooks are the tool of choice for many data scientists for their ability to tell a narrative with both code and natural language. These two types of discourse are contained within **cells** of the notebook, and we refer to these cells as either **code** cells or **markdown** cells ([markdown](https://en.wikipedia.org/wiki/Markdown) being the text formatting language used by Jupyter).

Your task is to predict the correct ordering of the cells in a given notebook whose markdown cells have been shuffled.

The notebooks in this dataset have been selected and processed to ensure their suitability for the competition task. All notebooks:

- Have been published publicly on Kaggle under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.
- Represent the most recently published version of the notebook.
- Contain at least one code cell and at least one markdown cell.
- Have code written in the Python language.
- Have had empty cells removed.

This is a code competition, in which you will submit a model or code that will be run against a future test set:

- The first-stage test set contains notebooks from an approximately 90-day historical window of time.
- The second-stage test set will contain a similar number of notebooks, collected from a future 90-day window of time. This is necessary to prevent models from looking up the order of existing public notebooks. The selection criteria for second-stage notebooks will be monitored for competition fairness purposes. For example, it will exclude competition participants' own notebooks.

## Training Data

- **train/** - A folder comprising about 140,000 JSON files with the filenames corresponding to the `id` field in the `csv` files. Each file contains the code and markdown cells of a Kaggle notebook. **The code cells are in their original (correct) order. The markdown cells have been shuffled** and placed after the code cells.
- **train_orders.csv** - Gives the correct order of the cells for each notebook in the `train/` folder.
    - `id` - The notebook in file `{id}.json`.
    - `cell_order` - A space delimited list of the correct cell ordering given in terms of the order in `{id}.json`.
- **train_ancestors.csv** - On Kaggle, a user may "fork" (that is, copy) the notebook of another user to create their own version. This file contains the forking history of notebooks in the training set. **Note: There is no corresponding file for the test set.**
    - `ancestor_id` - Identifies sets of notebooks that have a common origin or "ancestor". As no notebook in the test set has an ancestor in the training set, you may find this field to be of use as a grouping factor when constructing validation splits.
    - `parent_id` - Indicates that some version of the notebook `id` was forked from some version of the notebook `parent_id`. The notebook `parent_id` may or may not be present in the training data. (The parent may be missing because someone had forked a private notebook of their own, for instance.)

## Example Test Data

To help you author submission code, we include a few example instances selected from the test set. When you submit your notebook for scoring, this example data will be replaced by the actual test data, including the `sample_submission.csv` file.

- **test/** - A few example notebooks from the test set. The actual test set comprises about 20,000 notebooks in a format similar to the training set notebooks. No notebook in the test set has an ancestor in the training set.
- **sample_submission.csv** - A sample submission file in the correct format. See the [**Evaluation**](https://www.kaggle.com/competitions/AI4Code/overview/evaluation) page for more details.
