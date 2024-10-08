# Overview

## Description

Winning the fight against the COVID-19 pandemic will require an effective vaccine that can be equitably and widely distributed. Building upon decades of research has allowed scientists to accelerate the search for a vaccine against COVID-19, but every day that goes by without a vaccine has enormous costs for the world nonetheless. We need new, fresh ideas from all corners of the world. Could online gaming and crowdsourcing help solve a worldwide pandemic? Pairing scientific and crowdsourced intelligence could help computational biochemists make measurable progress.

mRNA vaccines have taken the lead as the fastest vaccine candidates for COVID-19, but currently, they face key potential limitations. One of the biggest challenges right now is how to design super stable messenger RNA molecules (mRNA). Conventional vaccines (like your seasonal flu shots) are packaged in disposable syringes and shipped under refrigeration around the world, but that is not currently possible for mRNA vaccines.

Researchers have observed that RNA molecules have the tendency to spontaneously degrade. This is a serious limitation--a single cut can render the mRNA vaccine useless. Currently, little is known on the details of where in the backbone of a given RNA is most prone to being affected. Without this knowledge, current mRNA vaccines against COVID-19 must be prepared and shipped under intense refrigeration, and are unlikely to reach more than a tiny fraction of human beings on the planet unless they can be stabilized.

![](https://storage.googleapis.com/kaggle-media/competitions/Stanford/banner%20(2).png)

The Eterna community, led by Professor Rhiju Das, a computational biochemist at Stanford's School of Medicine, brings together scientists and gamers to solve puzzles and invent medicine. Eterna is an online video game platform that challenges players to solve scientific problems such as mRNA design through puzzles. The solutions are synthesized and experimentally tested at Stanford by researchers to gain new insights about RNA molecules. The Eterna community has previously unlocked new scientific principles, made new diagnostics against deadly diseases, and engaged the world's most potent intellectual resources for the betterment of the public. The Eterna community has advanced biotechnology through its contribution in over 20 publications, including advances in RNA biotechnology.

In this competition, we are looking to leverage the data science expertise of the Kaggle community to develop models and design rules for RNA degradation. Your model will predict likely degradation rates at each base of an RNA molecule, trained on a subset of an Eterna dataset comprising over 3000 RNA molecules (which span a panoply of sequences and structures) and their degradation rates at each position. We will then score your models on a second generation of RNA sequences that have just been devised by Eterna players for COVID-19 mRNA vaccines. These final test sequences are currently being synthesized and experimentally characterized at Stanford University in parallel to your modeling efforts -- Nature will score your models!

Improving the stability of mRNA vaccines was a problem that was being explored before the pandemic but was expected to take many years to solve. Now, we must solve this deep scientific challenge in months, if not weeks, to accelerate mRNA vaccine research and deliver a refrigerator-stable vaccine against SARS-CoV-2, the virus behind COVID-19. The problem we are trying to solve has eluded academic labs, industry R&D groups, and supercomputers, and so we are turning to you. To help, you can join the team of video game players, scientists, and developers at Eterna to unlock the key in our fight against this devastating pandemic.

[![](https://storage.googleapis.com/kaggle-media/competitions/Stanford/logo_eterna_reverse.png)         ](http://eternagame.org/)[![](https://storage.googleapis.com/kaggle-media/competitions/Stanford/OpenVaccine-logo-black-transp-bg%20(1).png)         ](https://challenges.eternagame.org/)[![](https://storage.googleapis.com/kaggle-media/competitions/Stanford/logo_das.png)](http://daslab.stanford.edu/)

## Evaluation

Submissions are scored using MCRMSE, mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

From the `Data` page: There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

### Submission File

For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Timeline

- **October 2, 2020** - Entry deadline. You must accept the competition rules before this date in order to compete.

- **October 2, 2020** - Team Merger deadline. This is the last day participants may join or merge teams.

- **October 6, 2020** - Final submission deadline. *Date extended by one day for new data refresh.*

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - $ 12,000
- 2nd Place - $ 8,000
- 3rd Place - $ 5,000

## Additional Resources

**How RNA vaccines work, and their issues, featuring Dr. Rhiju Das**\
<https://www.pbs.org/wgbh/nova/video/rna-coronavirus-vaccine/>

**Launch of the OpenVaccine challenge**\
<https://scopeblog.stanford.edu/2020/05/20/stanford-biochemist-works-with-gamers-to-develop-covid-19-vaccine/>

**The impossibility of mass immunization with current RNA vaccines**\
<https://www.wsj.com/articles/from-freezer-farms-to-jets-logistics-operators-prepare-for-a-covid-19-vaccine-11598639012>

**CDC prepares for frozen mRNA vaccines**\
<https://www.cdc.gov/vaccines/acip/meetings/downloads/slides-2020-08/COVID-08-Dooling.pdf>

**Eterna, the crowdsourcing platform for RNA design where OpenVaccine's RNA sequences are coming from**\
<https://eternagame.org>

**How to build a better vaccine from the comfort of your own web browser**\
<https://medium.com/eternaproject/how-to-build-a-better-vaccine-from-the-comfort-of-your-own-web-browser-233343e0210d>

## Citation

Rhiju Das, H Wayment-Steele, Do Soon Kim, Christian Choe, Bojan Tunguz, Walter Reade, Maggie Demkin. (2020). OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction. Kaggle. https://kaggle.com/competitions/stanford-covid-vaccine

# Dataset Description

In this competition, you will be predicting the degradation rates at various locations along RNA sequence.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: `reactivity`, `deg_Mg_pH10`, and `deg_Mg_50C`.

### Files

- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns

- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

### Additional Notes

At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.
