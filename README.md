# MLE-bench

Code for the paper ["MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering"](https://arxiv.org/abs/2410.07095). We have released the code used to construct the dataset, the evaluation logic, as well as the agents we evaluated for this benchmark.

## Results

The table below shows the scores from the paper of models on MLE-Bench Lite and MLE-Bench, on the headline metric of 'percentage of submissions where the model achieved any medal'.

| Model                      | MLE-Bench Lite (%) | MLE-Bench (%) |
| -------------------------- | -----------------: | ------------: |
| **AIDE**                   |                    |               |
| o1-preview                 |         34.3 ± 2.4 |    16.9 ± 1.1 |
| gpt-4o-2024-08-06          |         19.0 ± 1.3 |     8.6 ± 0.5 |
| llama-3.1-405b-instruct    |          8.3 ± 2.6 |     3.1 ± 0.9 |
| claude-3-5-sonnet-20240620 |         19.4 ± 4.9 |     7.5 ± 1.8 |
| **MLAB**                   |                    |               |
| gpt-4o-2024-08-06          |          4.2 ± 1.5 |     1.3 ± 0.5 |
| **OpenHands**              |                    |               |
| gpt-4o-2024-08-06          |         11.5 ± 3.4 |     5.1 ± 1.3 |

## Setup

Some MLE-bench competition data is stored using [Git-LFS](https://git-lfs.com/).
Once you have downloaded and installed LFS, run:

```console
git lfs fetch --all
git lfs pull
```

You can install `mlebench` with pip:

```console
pip install -e .
```

### Pre-Commit Hooks (Optional)

If you're committing code, you can install the pre-commit hooks by running:

```console
pre-commit install
```

## Dataset

The MLE-bench dataset is a collection of 75 Kaggle competitions which we use to evaluate the ML engineering capabilities of AI systems.

Since Kaggle does not provide the held-out test set for each competition, we
provide preparation scripts that split the publicly available training set into
a new training and test set.

For each competition, we also provide grading scripts that can be used to
evaluate the score of a submission.

We use the [Kaggle API](https://github.com/Kaggle/kaggle-api) to download the
raw datasets. Ensure that you have downloaded your Kaggle credentials
(`kaggle.json`) and placed it in the `~/.kaggle/` directory (this is the default
location where the Kaggle API looks for your credentials). To download and prepare the MLE-bench dataset, run the following, which will download and prepare the dataset in your system's default cache directory. Note, we've found this to take two days when running from scratch:

```console
mlebench prepare --all
```

Alternatively, you can prepare the dataset for a specific competition by
running:

```console
mlebench prepare -c <competition-id>
```

Run `mlebench prepare --help` to see the list of available competitions.

### MLE-bench Lite

We also provide a lite version of the MLE-bench dataset, which contains all the 22 competitions from the Low complexity category. To prepare the lite dataset, run:

```console
mlebench prepare --lite
```

The Lite dataset contains the following competitions:

|Competition ID|Category                                       |Dataset Size (GB)     |
|--------------|-----------------------------------------------|--------------------|
|aerial-cactus-identification|Image Classification                           |0.0254              |
|aptos2019-blindness-detection|Image Classification                           |10.22               |
|denoising-dirty-documents|Image To Image                                 |0.06                |
|detecting-insults-in-social-commentary|Text Classification                            |0.002               |
|dog-breed-identification|Image Classification                           |0.75                |
|dogs-vs-cats-redux-kernels-edition|Image Classification                           |0.85                |
|histopathologic-cancer-detection|Image Regression                               |7.76                |
|jigsaw-toxic-comment-classification-challenge|Text Classification                            |0.06                |
|leaf-classification|Image Classification                           |0.036               |
|mlsp-2013-birds|Audio Classification                           |0.5851              |
|new-york-city-taxi-fare-prediction|Tabular                                        |5.7                 |
|nomad2018-predict-transparent-conductors|Tabular                                        |0.00624             |
|plant-pathology-2020-fgvc7|Image Classification                           |0.8                 |
|random-acts-of-pizza|Text Classification                            |0.003               |
|ranzcr-clip-catheter-line-classification|Image Classification                           |13.13               |
|siim-isic-melanoma-classification|Image Classification                           |116.16              |
|spooky-author-identification|Text Classification                            |0.0019              |
|tabular-playground-series-dec-2021|Tabular                                        |0.7                 |
|tabular-playground-series-may-2022|Tabular                                        |0.57                |
|text-normalization-challenge-english-language|Seq->Seq                                       |0.01                |
|text-normalization-challenge-russian-language|Seq->Seq                                       |0.01                |
|the-icml-2013-whale-challenge-right-whale-redux|Audio Classification                           |0.29314             |


## Grading Submissions

Answers for competitions must be submitted in CSV format; the required format is described in each competition's description, or shown in a competition's sample submission file. You can grade multiple submissions by using the `mlebench grade` command. Given a JSONL file, where each line corresponds with a submission for one competition, `mlebench grade` will produce a grading report for each competition. The JSONL file must contain the following fields:
- `competition_id`: the ID of the competition in our dataset.
- `submission_path`: a `.csv` file with the predictions for the specified
  competition.

See more information by running `mlebench grade --help`.

You can also grade individual submissions using the `mlebench grade-sample` command. For example, to grade a submission for the Spaceship Titanic competition, you can run:

```console
mlebench grade-sample <PATH_TO_SUBMISSION> spaceship-titanic
```

See more information by running `mlebench grade-sample --help`.

## Environment

We provide a base Docker image `mlebench-env` which is the base environment for our agents. This base image contains:
- Conda environment used to execute our agents. We optionally (default true) install Python packages in this environment which are commonly used across our agents. If you don't want to install these packages, set the `INSTALL_HEAVY_DEPENDENCIES` environment variable to `false` when building the image, by adding `--build-arg INSTALL_HEAVY_DEPENDENCIES=false` to the `docker build` command below
- Instructions for agents to follow when creating their submission
- Grading server for agents to use when checking that the structure of their submission is correct

Build this image by running:

```bash
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
```

## Agents

We purposefully designed our benchmark to not make any assumptions about the agent that produces submissions, so agents can more easily be evaluated on this benchmark. We evaluated three open-source agents; we discuss this procedure in [agents/README.md](agents/README.md).

## Extras

We include additional features in the MLE-bench repository that may be useful
for MLE-bench evaluation. These include a rule violation detector and
a plagiarism detector. We refer readers to
[extras/README.md](extras/README.md) for more information.

## Examples

We collect example usage of this library in the `examples/` directory, see [examples/README.md](examples/README.md) for more information.

## Experiments

We place the code specific to the experiments from our publication of the
benchmark in the `experiments/` directory:
- For instance, our competition splits are available in `experiments/splits/`.
- For a completed set of runs from a given agent, you can use the provided
`experiments/make_submission.py` script to compile its submission for grading.
- We release our methodology for the "familiarity" experiments in `experiments/familiarity/`, see [experiments/familiarity/README.md](experiments/familiarity/README.md) for more information.

## Dev

Note, when running `pytest` locally, be sure to accept the competition rules otherwise the tests will fail.

## Authors

Chan Jun Shern, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, Aleksander Mądry

## Citation

Please cite using the following BibTeX entry:
```
@article{chan2024mle-bench,
  title={MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering},
  author={Jun Shern Chan and Neil Chowdhury and Oliver Jaffe and James Aung and Dane Sherburn and Evan Mays and Giulio Starace and Kevin Liu and Leon Maksin and Tejal Patwardhan and Lilian Weng and Aleksander Mądry},
  year={2024},
  eprint={2410.07095},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.07095}
}
```
