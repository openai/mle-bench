# MLE-bench -- Runs

This directory contains the grading reports for MLE-bench experiment runs.

Each subdirectory in this directory corresponds to a "run group". A run group is
nothing more but a collection of run results across multiple competitions and
seeds. Here we share the run group's grading_report JSON file, which contains
the scores achieved. A run group will belong to a particular experiment, with
any given experiment potentially having multiple run groups associated with it.

An experiment here is typically a specific agent setup, e.g. gpt-4o with AIDE
scaffolding.

So, for example, to find the overall performance of gpt-4o with AIDE
scaffolding, you would find all run groups associated with that experiment, read
the competition scores and aggregate across all the seeds.

In cases where there are more seeds for a competition than reported, this is due
to earlier seeds for the competition failing due to hardware issues. In this
case and we took the final _N_ seeds for computing the result, where _N_ is the
number of seeds for the competition with the least number of seeds in the
experiment.

We refer readers to `runs/run_group_experiments.csv` for a mapping from
experiment id to run groups. Experiment id descriptions can be found in the
table below.

| **experiment_id**                   | **notes**                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------|
| biggpu-gpt4o-aide                   | GPT-4o (AIDE), Extra GPU (two 24GB A10 GPUs rather than one)               |
| cpu-gpt4o-aide                      | GPT-4o (AIDE), CPU-only (no GPU access).                                   |
| extratime-gpt4o-aide                | GPT-4o (AIDE) with 100 hours of time (rather than 24)                      |
| models-claude35sonnet-aide          | Claude 3.5 Sonnet on AIDE scaffolding                                      |
| models-llama-3.1-405B-instruct-aide | LLama 3.1 405B Instruct on AIDE scaffolding                                |
| models-o1-preview-aide              | o1-preview on AIDE scaffolding                                             |
| obfuscation-gpt4o-aide              | GPT-4o (AIDE) with obfuscated descriptions                                 |
| scaffolding-gpt4o-aide              | GPT-4o on AIDE scaffolding                                                 |
| scaffolding-gpt4o-mlab              | GPT-4o on MLAB scaffolding                                                 |
| scaffolding-gpt4o-opendevin         | GPT-4o on OpenDevin scaffolding                                            |
| o1-preview-R&D-Agent                 | o1-preview on R&D-Agent scaffolding, 12 vCPUs, 220GB of RAM, and 1 V100 GPU |
| deepseek-r1-ML-Master               | Deepseek-R1 on ML-Master scaffolding, 12 hours, 36 vCPUs, 512GB of RAM, and 1 A100 GPU|
| multi-agent-Neo                     | Multi-Agent Ensemble of LLMs (GPT 4.1 and Claude Sonnet 4.0) on NEO scaffolding, 36 hours, 24 vCPUs, 144GB of RAM, and 1 A100 GPU|
| o3-gpt-4.1-R&D-Agent                 | O3 as researcher and gpt-4.1 as developer on R&D-Agent scaffolding, 12 vCPUs, 220GB of RAM, and 1 V100 GPU |
| deepseek-r1-InternAgent              | Deepseek-R1 on InternAgent scaffolding, 12 hours, 32 vCPUs, 230GB of RAM, and 1 A800 GPU|
