## Description

Can you help detect toxic comments ― *and* minimize unintended model bias? That's your challenge in this competition.

The Conversation AI team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both part of Alphabet), builds technology to protect voices in conversation. A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything *rude, disrespectful or otherwise likely to make someone leave a discussion*.

Last year, in the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description), you built multi-headed models to recognize toxicity and several subtypes of toxicity. This year's competition is a related challenge: building toxicity models that operate fairly across a diverse range of conversations.

Here's the background: When the Conversation AI team first built toxicity models, they found that the models [incorrectly learned to associate](https://medium.com/the-false-positive/unintended-bias-and-names-of-frequently-targeted-groups-8e0b81f80a23) the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. Develop strategies to reduce unintended bias in machine learning models, and you'll help the Conversation AI team, and the entire industry, build models that work well for a wide range of conversations.

*Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.*

### Acknowledgments

The Conversation AI team would like to thank Civil Comments for making this dataset available publicly and the [Online Hate Index Research Project](http://dh.berkeley.edu/projects/online-hate-index-ohi-research-project?utm_source=BCNM+Subscribers&utm_campaign=d5d78bba5e-natasha-schull-oct12_COPY_01&utm_medium=email&utm_term=0_eb59bfff9e-d5d78bba5e-281420185) at [D-Lab](https://dlab.berkeley.edu/), University of California, Berkeley, whose labeling survey/instrument informed the dataset labeling. We'd also like to thank everyone who has contributed to Conversation AI's research, especially those who took part in our last competition, the success of which led to the creation of this challenge.

## Evaluation

### Competition Evaluation

This competition will use a newly developed metric that combines several submetrics to balance overall performance with various aspects of unintended bias.

First, we'll define each submetric.

### Overall AUC

This is the ROC-AUC for the full evaluation set.

### Bias AUCs

To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. You can learn more about these metrics in Conversation AI's recent paper *[Nuanced Metrics for Measuring Unintended Bias with Real Data in Text Classification](https://arxiv.org/abs/1903.04561)*.

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

### Generalized Mean of Bias AUCs

To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

### Final Metric

We combine the overall AUC with the generalized mean of the Bias AUCs to calculate the final model score:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

**While the leaderboard will be determined by this single number, we highly recommend looking at the individual submetric results, [as shown in this kernel](https://www.kaggle.com/dborkan/benchmark-kernel), to guide you as you develop your models.**

### Submission File

```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Prizes

We expect there to be a large variety of ways to mitigate bias and, like other Kaggle competitions, we expect tight competition. We also recognize that bias is a complex and nuanced idea that can only be measured imperfectly. Our goal is for competitors to develop, evaluate, and compare as many bias mitigation strategies as possible, and so we'll be rewarding as many of them as we can.

- 1st Place - $12,000
- 2nd Place - $10,000
- 3rd Place - $8,000
- 4th Place - $5,000
- 5th Place - $5,000
- 6th Place - $5,000
- 7th Place - $5,000
- 8th Place - $5,000
- 9th Place - $5,000
- 10th Place - $5,000

## Timeline

- **June 19, 2019** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **June 19, 2019** - Team Merger deadline. This is the last day participants may join or merge teams.
- **June 26, 2019** - Final submission deadline. After this date, we will not be taking any more submissions. Remember to select your two best submissions to be rescored during the re-run period.
- **June 27 to July 19, 2019** - Kernel Re-runs on Private Test Set.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## FAQ

### How does this metric compare with other measures of bias and fairness?

There are a wide range of fairness criteria and bias measurement metrics available in the literature. The metric used here is similar to the Equality of Odds criteria presented in *[Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413.pdf)* and other related metrics, in that it relies on having labeled test data broken down across the different identity groups under consideration and in that it allows for there to be a relationship between the identity groups and the concept being modeled. We believe this is a good fit for toxicity classification because we do expect there to be differences in the amount and severity of toxicity directed at different groups. Our metrics differ from other similar metrics in that they are threshold-agnostic, meaning that we evaluate on the raw model scores themselves, rather than converting the model into a binary threshold before evaluation. Please see our paper *[Nuanced Metrics for Measuring Unintended Bias with Real Data in Text Classification](https://arxiv.org/abs/1903.04561)* for more details on the bias metrics.

### How can I calculate the individual Bias AUCs for my models?

Please see the example in our [Benchmark Kernel](https://www.kaggle.com/dborkan/benchmark-kernel) to calculate the individual Bias AUCs. We highly recommend looking at these values regularly as you develop your models.

### What are the benchmarks for this competition?

We provide three benchmarks for this competition:

- **Benchmark Kernel:** A simple CNN model provided as an example submission in the [Benchmark Kernel](https://www.kaggle.com/dborkan/benchmark-kernel).
- **TOXICITY@1 Benchmark:** Perspective API's earliest toxicity model, launched in February 2017.
- **TOXICITY@6 Benchmark:** Perspective API's current toxicity model, launched in August 2018.

The Perspective toxicity model has undergone several bias mitigation steps between TOXICITY@1 and TOXICITY@6, which you can learn about in Jigsaw's two blog posts on [unintended bias](https://medium.com/the-false-positive/unintended-bias-and-names-of-frequently-targeted-groups-8e0b81f80a23) and [transparency](https://medium.com/the-false-positive/increasing-transparency-in-machine-learning-models-311ee08ca58a). As shown in *[Nuanced Metrics for Measuring Unintended Bias with Real Data in Text Classification](https://arxiv.org/abs/1903.04561)*, these changes focused on reducing false positive bias on short comments (BPSN AUC), where unintended bias was strongest in TOXICITY@1. On the full Civil Comments set, which evaluates a range of comment lengths, we see less difference between these two models. We look forward to seeing competitors further improve the model and reduce unintended bias beyond these benchmarks.

### How was the data collected for this challenge?

The comments in this challenge come from an archive of the [Civil Comments](https://medium.com/@aja_15265/saying-goodbye-to-civil-comments-41859d3a2b1d) platform, a commenting plugin for independent news sites. These public comments were created from 2015 - 2017 and appeared on approximately 50 English-language news sites across the world. When Civil Comments shut down in 2017, they chose to make the public comments available in a lasting open archive to enable future research. The original data, published on [figshare](https://figshare.com/articles/data_json/7376747), includes the public comment text, some associated metadata such as article IDs, timestamps and commenter-generated "civility" labels, but does not include user ids. Jigsaw extended this dataset by adding additional labels for toxicity and identity mentions.

### Who labeled the data? How were the labels verified?

Toxicity and identity mentions were labeled via the crowd rating platform [Figure Eight](https://www.figure-eight.com/). Label quality was enforced using test questions, a system where approximately 10% of the comments raters see already have known correct labels. Raters who miss too many of these labels are disqualified from rating. We recognize that toxicity and identity can be subjective, so we collect labels from up to 10 people per comment, to capture a range of opinions.

### Why were these particular identities selected?

This set of identities was designed to balance the coverage of identities, crowd rater accuracy, and ensure that each labeled identity has enough examples in the final data set to provide meaningful results.

### Why are only some of the comments labeled for identity references?

When designing any human rating task, it's important that the concept raters are looking for (whether it's toxicity, identities, or anything else) appears with some level of frequency, to keep the raters engaged. With this data, we used a combination of model predictions and word matching to find approximately 250,000 comments that were likely to contain identity references. We combined that data with ~250,000 randomly selected comments to generate a set of ~500,000 to be labeled, where we could expect about 50% of comments to have identity references.

### Will the data sets in this competition be released after the competition finishes?

Yes! The full dataset, including the test set and all labels, will be released under a Creative Commons license after the competition ends. The comments from the training set and associated metadata from Civil Comments are already released on [figshare](https://figshare.com/articles/data_json/7376747).

### Can I use external datasets?

You may use any additional datasets, as long as they are appropriately licensed to be used for this purpose and shared publicly. Please post a link to your external dataset(s) on the [official external data forum thread](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/87235).

### Can I add additional labels the training data?

Feel free! Please post a link to your newly labeled dataset on the [official external data forum thread](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/87235).

## Kernels Requirements

### This is a Kernels-only competition

Submissions to this competition must be made through Kernels. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- CPU Kernel <= 9 hours run-time
- GPU Kernel <= 2 hours run-time
- No internet access enabled
- External data, freely & publicly available, is allowed, including pre-trained models
- No use of other kernels as inputs into your submission kernel. This includes utility scripts.
- No custom packages enabled in kernels
- Submission file must be named "submission.csv"

Please see the [Kernels-only FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ) for more information on how to submit.

## Citation

cjadams, Daniel Borkan, inversion, Jeffrey Sorensen, Lucas Dixon, Lucy Vasserman, nithum. (2019). Jigsaw Unintended Bias in Toxicity Classification. Kaggle. https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification

# **Dataset Description**

*Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.*

## Background

At the end of 2017 the [Civil Comments](https://medium.com/@aja_15265/saying-goodbye-to-civil-comments-41859d3a2b1d) platform shut down and chose make their ~2m public comments from their platform available in a lasting open archive so that researchers could understand and improve civility in online conversations for years to come. Jigsaw sponsored this effort and extended annotation of this data by human raters for various toxic conversational attributes.

In the data supplied for this competition, the text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. **Only identities with more than 500 examples in the test set (combined public and private) will be included in the evaluation calculation. These identities are shown in bold.**

- **male**
- **female**
- transgender
- other_gender
- heterosexual
- **homosexual_gay_or_lesbian**
- bisexual
- other_sexual_orientation
- **christian**
- **jewish**
- **muslim**
- hindu
- buddhist
- atheist
- other_religion
- **black**
- **white**
- asian
- latino
- other_race_or_ethnicity
- physical_disability
- intellectual_or_learning_disability
- **psychiatric_or_mental_illness**
- other_disability

Note that the data contains different comments that can have the exact same text. Different comments that have the same text may have been labeled with different targets or subgroups.

## Examples

Here are a few examples of comments and their associated toxicity and identity labels. Label values range from 0.0 - 1.0 represented the fraction of raters who believed the label fit the comment.

**Comment**: *i'm a white woman in my late 60's and believe me, they are not too crazy about me either!!*

- **Toxicity Labels:** All 0.0
- **Identity Mention Labels:** female: 1.0, white: 1.0 (all others 0.0)

**Comment**: *Why would you assume that the nurses in this story were women?*

- **Toxicity Labels:** All 0.0
- **Identity Mention Labels:** female: 0.8 (all others 0.0)

**Comment**: *Continue to stand strong LGBT community. Yes, indeed, you'll overcome and you have.*

- **Toxicity Labels:** All 0.0
- **Identity Mention Labels:** homosexual_gay_or_lesbian: 0.8, bisexual: 0.6, transgender: 0.3 (all others 0.0)

In addition to the labels described above, the dataset also provides metadata from Jigsaw's annotation: `toxicity_annotator_count` and `identity_annotator_count`, and metadata from Civil Comments: `created_date`, `publication_id`, `parent_id`, `article_id`, `rating`, `funny`, `wow`, `sad`, `likes`, `disagree`. Civil Comments' label `rating` is the civility rating Civil Comments users gave the comment.

## Labelling Schema

To obtain the toxicity labels, each comment was shown to up to 10 annotators*. Annotators were asked to: "Rate the toxicity of this comment"

- Very Toxic (a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion or give up on sharing your perspective)
- Toxic (a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion or give up on sharing your perspective)
- Hard to Say
- Not Toxic

These ratings were then aggregated with the `target` value representing the fraction of annotations that annotations fell within the former two categories.

To collect the identity labels, annotators were asked to indicate all identities that were mentioned in the comment. An example question that was asked as part of this annotation effort was: "What genders are mentioned in the comment?"

- Male
- Female
- Transgender
- Other gender
- No gender mentioned

Again, these were aggregated into fractional values representing the fraction of raters who said the identity was mentioned in the comment.

The distributions of labels and subgroup between Train and Test can be assumed to be similar, but not exact.

- Note: Some comments were seen by many more than 10 annotators (up to thousands), due to sampling and strategies used to enforce rater accuracy.

## Annotators

Toxicity and identity labeling was done in 2018 on the Figure Eight crowd rating platform, which has since been purchased by Appen. Annotators came from all over the world, and all were proficient in English.

Raters were compensated 1.5 cents per judgment, a rate that was set based on two factors: what rates were competitive on the platform and targeting an hourly wage appropriate for raters' locales.

Figure Eight offers raters the option to select from multiple tasks with different pay, so rates need to be competitive to attract enough raters. At a rate of 1.5 cents per judgment, enough raters participated to complete this annotation in a few weeks. Following task completion, raters were given a satisfaction survey where the average score for "Pay" was 3.8 out of 5, which aligned with Figure Eight's recommendation to target pay to a satisfaction score greater than 3.5.

Hourly rates for workers depend on how fast judgements were completed. Most raters will have earned between $0.90/hour (at one comment per minute) to $5.40/hour (at 6 comments per minute), which aligns with typical hourly pay in the geographic regions where most raters are located.

Since this dataset was annotated in 2018, more [tools](https://success.appen.com/hc/en-us/articles/9557008940941-Guide-to-Fair-Pay) have become available to help set rates. Consequently, typical pay for data annotation is increasing globally.

## File descriptions

- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

## Usage

This dataset is released under [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/), as is the [underlying comment text](https://figshare.com/articles/data_json/7376747).