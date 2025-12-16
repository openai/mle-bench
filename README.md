# MLE-bench

Code for the paper ["MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering"](https://arxiv.org/abs/2410.07095). We have released the code used to construct the dataset, the evaluation logic, as well as the agents we evaluated for this benchmark.

## Leaderboard

| Agent | LLM(s) used | Low == Lite (%) | Medium (%) | High (%) | All (%) | Running Time (hours) | Date | Grading Reports Available | Source Code Available |
|-------|-------------|-----------------|------------|----------|---------|----------------------|------|---------------------------|----------------------|
| [ML-Master 2.0](https://github.com/sjtu-sai-agents/ML-Master) | Deepseek-V3.2-Speciale | 75.76 ± 1.51 | 50.88 ± 3.51 | 42.22 ± 2.22 | 56.44 ± 2.47 | 24 | 2025-12-16 | ✓ | X |
| [Leeroo](https://leeroo.com/) | Gemini-3-Pro-Preview[^3] | 68.18 ± 2.62[^2] | 44.74 ± 1.52[^2] | 40.00 ± 0.00[^2] | 50.67 ± 1.33[^2] | 24 | 2025-12-07 | ✓ | X |
| [Thesis](https://thesislabs.ai) | gpt-5-codex | 65.15 ± 1.52 | 45.61 ± 7.18 | 31.11 ± 2.22 | 48.44 ± 3.64 | 24 | 2025-11-10 | ✓ | X |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro-1.5 | Gemini-2.5-Pro | 68.18 ± 2.62 | 34.21 ± 1.52  |  33.33 ±  0.00 |  44.00 ± 1.33 | 24 | 2025-11-25 | ✓ | X |
| [FM Agent](https://github.com/baidubce/FM-Agent) | Gemini-2.5-Pro | 62.12 ± 1.52 | 36.84 ± 1.52 | 33.33 ± 0.00 | 43.56 ± 0.89 | 24 | 2025-10-10 | ✓ | X |
| [Operand](https://operand.com) ensemble | gpt-5 (low verbosity/effort)[^1] | 63.64 ± 0.00 | 33.33 ± 0.88[^2] | 20.00 ± 0.00[^2] | 39.56 ± 0.44[^2] | 24 | 2025-10-06 | ✓ | X |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro-1.0 | Gemini-2.5-Pro | 66.67 ± 1.52  | 25.44  ±  0.88 | 31.11 ± 2.22  | 38.67 ± 0.77 | 12 | 2025-11-03 | ✓ | X |
| [InternAgent](https://github.com/Alpha-Innovator/InternAgent/) | deepseek-r1 | 62.12 ± 3.03 | 26.32 ± 2.63 | 24.44 ± 2.22| 36.44 ± 1.18 | 12 | 2025-09-12 | ✓ | X |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | gpt-5 | 68.18 ± 2.62 | 21.05 ± 1.52 | 22.22 ± 2.22 | 35.11 ± 0.44 | 12 | 2025-09-26 | ✓ | ✓ |
| [Neo](https://heyneo.so/) multi-agent | undisclosed | 48.48 ± 1.52 | 29.82 ± 2.32 | 24.44 ± 2.22 | 34.22 ± 0.89 | 36 | 2025-07-28 | ✓ | X |
| [AIRA-dojo](https://github.com/facebookresearch/aira-dojo/) | o3 | 55.00 ± 1.47 | 21.97 ± 1.17 | 21.67 ± 1.07 | 31.60 ± 0.82 | 24 | 2025-05-15 | ✓ | ✓ |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o3 + GPT-4.1 | 51.52 ± 4.01 | 19.30 ± 3.16 | 26.67 ± 0.00 | 30.22 ± 0.89 | 24 | 2025-08-15 | ✓ | ✓ |
| [ML-Master](https://github.com/zeroxleo/ML-Master) | deepseek-r1 | 48.48 ± 1.52 | 20.18 ± 2.32 | 24.44 ± 2.22 | 29.33 ± 0.77 | 12 | 2025-06-17 | ✓ | ✓ |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o1-preview | 48.18 ± 1.11 | 8.95 ± 1.05 | 18.67 ± 1.33 | 22.40 ± 0.50 | 24 | 2025-05-14 | ✓ | ✓ |
| [AIDE](https://github.com/wecoai/aideml) | o1-preview | 35.91 ± 1.86 | 8.45 ± 0.43 | 11.67 ± 1.27 | 17.12 ± 0.61 | 24 | 2024-10-08 | ✓ | ✓ |
| [AIDE](https://github.com/wecoai/aideml) | gpt-4o-2024-08-06 | 18.55 ± 1.26 | 3.06 ± 0.33 | 8.15 ± 0.84 | 8.63 ± 0.54 | 24 | 2024-10-08 | ✓ | ✓ |
| [AIDE](https://github.com/wecoai/aideml) | claude-3-5-sonnet-20240620 | 19.70 ± 1.52 | 2.63 ± 1.52 | 2.22 ± 2.22 | 7.56 ± 1.60 | 24 | 2024-10-08 | ✓ | ✓ |
| OpenHands | gpt-4o-2024-08-06 | 12.12 ± 1.52 | 1.75 ± 0.88 | 2.22 ± 2.22 | 4.89 ± 0.44 | 24 | 2024-10-08 | ✓ | ✓ |
| [AIDE](https://github.com/wecoai/aideml) | llama-3.1-405b-instruct | 10.23 ± 1.14 | 0.66 ± 0.66 | 0.00 ± 0.00 | 3.33 ± 0.38 | 24 | 2024-10-08 | ✓ | ✓ |
| MLAB | gpt-4o-2024-08-06 | 4.55 ± 0.86 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.60 ± 0.27 | 24 | 2024-10-08 | ✓ | ✓ |

[^1]: With some light assistance from an ensemble of models including
    Gemini-2.5-Pro, Grok-4, and Claude 4.1 Opus, distilled by Gemini-2.5-Pro.
[^2]: Computed by padding incomplete seeds with failing scores.
[^3]: The architecture is primarily driven by Gemini-3-Pro-Preview, with a subset of modules utilizing GPT-5 and GPT-5-mini.
### Producing Scores for the Leaderboard

To produce the scores for the leaderboard, please organize your grading reports
in the `runs/` folder organized by run groups, with one grading report per run
group. Identify the run groups for your submission in
`runs/run_group_experiments.csv` with an experiment id. Then run

```bash
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split low
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split medium
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split high
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split split75
```

Report the mean and standard error of the mean (SEM) for each of the splits on
the reported `any_medal_percentage` metric. The `--split75` flag corresponds to
the `All (%)` column.

## Benchmarking

This section describes a canonical setup for comparing scores on MLE-bench. We recommend the following:
- Repeat each evaluation with at least 3 seeds and report the Any Medal (%) score as the mean ± one standard error of the mean. The evaluation (task and grading) itself is deterministic, but agents/LLMs can be quite high-variance!
- Agent resources - not a strict requirement of the benchmark but please report if you stray from these defaults!
  - Runtime: 24 hours
  - Compute: 36 vCPUs with 440GB RAM and one 24GB A10 GPU
- Include a breakdown of your scores across Low, Medium, High, and All complexity [splits](experiments/splits) (see *Lite evaluation* below for why this is useful).

### Lite Evaluation

Evaluating agents with the above settings on the full 75 competitions of MLE-bench can be expensive. For users preferring a "lite" version of the benchmark, we recommend using the [Low complexity split](https://github.com/openai/mle-bench/blob/main/experiments/splits/low.txt) of our dataset, which consists of only 22 competitions. This reduces the number of runs substantially, while still allowing fair comparison along one column of the table above.

Furthermore, the Low complexity competitions tend to be significantly more lightweight (158GB total dataset size compared to 3.3TB for the full set), so users may additionally consider reducing the runtime or compute resources available to the agents for further cost reduction. However, note that doing so risks degrading the performance of your agent. For example, see [Section 3.3 and 3.4 of our paper](https://arxiv.org/abs/2410.07095) where we have experimented with varying resources on the full competition set.

The Lite dataset contains the following competitions:

| Competition ID                              | Category                   | Dataset Size (GB) |
|---------------------------------------------|----------------------------|--------------------|
| aerial-cactus-identification                | Image Classification       | 0.0254            |
| aptos2019-blindness-detection               | Image Classification       | 10.22             |
| denoising-dirty-documents                   | Image To Image             | 0.06              |
| detecting-insults-in-social-commentary      | Text Classification        | 0.002             |
| dog-breed-identification                    | Image Classification       | 0.75              |
| dogs-vs-cats-redux-kernels-edition          | Image Classification       | 0.85              |
| histopathologic-cancer-detection            | Image Regression           | 7.76              |
| jigsaw-toxic-comment-classification-challenge | Text Classification        | 0.06              |
| leaf-classification                         | Image Classification       | 0.036             |
| mlsp-2013-birds                             | Audio Classification       | 0.5851            |
| new-york-city-taxi-fare-prediction          | Tabular                   | 5.7               |
| nomad2018-predict-transparent-conductors    | Tabular                   | 0.00624           |
| plant-pathology-2020-fgvc7                  | Image Classification       | 0.8               |
| random-acts-of-pizza                        | Text Classification        | 0.003             |
| ranzcr-clip-catheter-line-classification    | Image Classification       | 13.13             |
| siim-isic-melanoma-classification           | Image Classification       | 116.16            |
| spooky-author-identification                | Text Classification        | 0.0019            |
| tabular-playground-series-dec-2021          | Tabular                   | 0.7               |
| tabular-playground-series-may-2022          | Tabular                   | 0.57              |
| text-normalization-challenge-english-language | Seq->Seq                 | 0.01              |
| text-normalization-challenge-russian-language | Seq->Seq                 | 0.01              |
| the-icml-2013-whale-challenge-right-whale-redux | Audio Classification     | 0.29314           |

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

To prepare the lite dataset, run:

```console
mlebench prepare --lite
```

Alternatively, you can prepare the dataset for a specific competition by
running:

```console
mlebench prepare -c <competition-id>
```

Run `mlebench prepare --help` to see the list of available competitions.



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

## Known Issues

There are some known issues with certain MLE-bench competitions. Since we have
already received leaderboard submissions, we are postponing fixes to avoid
invalidating the leaderboard. Instead, we plan to release batched fixes in the
upcoming v2 release of MLE-bench on the
[openai/frontier-evals](https://github.com/openai/frontier-evals) repo, which will
include a version column in the leaderboard to distinguish between v1 and v2 results.
If you wish to make a submission to v1 in the meantime, please still include
the following competitions in your overall scores. The known issues are
catalogued below:

- **tensorflow-speech-recognition-challenge**:
  - The prepare.py script incorrectly prepares the test set such that there is a
    much larger range of test labels than there should be.
    [#63](https://github.com/openai/mle-bench/issues/63)
  - The prepare.py script does not properly create a test set where the speaker
    IDs are disjoint from those in train/val.
- **icecube-neutrinos-in-deep-ice**: Checksums are mismatch.
  [#58](https://github.com/openai/mle-bench/issues/58)
- **ranzcr-clip-catheter-line-classification**: The prepare.py script results in
  missing columns in the sample submission.
  [#30](https://github.com/openai/mle-bench/issues/30)
- **tabular-playground-series-dec-2021**: The leaderboard is crowded -- very
  little difference between the top score and the median score.
- **tabular-playground-series-may-2022**: The leaderboard is crowded -- very
  little difference between the top score and the median score.
- **jigsaw-toxic-comment-classification-challenge**: The leaderboard is crowded -- very
  little difference between the top score and the median score.
- **champs-scalar-coupling**: test molecules are missing in structures.csv.
  [#70](https://github.com/openai/mle-bench/pull/70)
- **multi-modal-gesture-recognition**: public test `.mat` files leak test labels.
  [#77](https://github.com/openai/mle-bench/issues/77)
- **smartphone-decimeter-2022**: The public test `span_log.nmea` files leak
  information that makes achieving a perfect score trivial.
  [#93](https://github.com/openai/mle-bench/issues/93)
- **hubmap-kidney-segmentation**: The public test `{image_id}.json` files leak
  information that makes achieving a close-to-perfect score trivial. They should
  be removed.
- **random-acts-of-pizza**: The field `giver_username_if_known` leaks the outcome,
  enabling trivial perfect prediction. This competition should be dropped.
  [#108](https://github.com/openai/mle-bench/issues/108)

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
