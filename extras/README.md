# MLE-bench Extras

In this folder we provide additional features not included in the core MLE-bench
package, that may nevertheless prove to be useful for evaluation.

## Rule Violation Detection

We offer a rule violation tool to run on the logs and code produced by agents
when creating submissions to individual competitions for the eval. The purpose
of this tool is to flag potential cheating or other issues in how the agent
created the submission, through the use of `gpt4o-mini`.

```console
usage: python rule_violation_detector/run.py [-h] --submission SUBMISSION --output-dir OUTPUT_DIR
              [--questions QUESTIONS [QUESTIONS ...]]
              [--extra-prompt EXTRA_PROMPT]

Script for analyzing submissions and detecting plagiarism.

options:
  -h, --help            show this help message and exit
  --submission SUBMISSION
                        Path to the JSONL file of submissions. Format mimics
                        README.md#submission-format, with `logs_path` and/or
                        `code_path` instead of `submission_path`.
  --output-dir OUTPUT_DIR
                        Path to the directory where the analysis CSV will be
                        saved.
  --questions QUESTIONS [QUESTIONS ...]
                        List of question IDs to ask for each log file defined
                        in `prompts.py`. Defaults to all questions.
  --extra-prompt EXTRA_PROMPT
                        Path to plaintext file containing additional prompt to
                        be concatenated to end of the base analysis prompt.
                        Useful for e.g. different log formats across agents.

```

For each code or log file pointed to in the submission JSONL file, this will ask
gpt4o-mini a series of question to determine if the submission has violated a
particular set of rules, as defined in `prompts.py`.

## Plagiarism Detection

We offer a plagiarism detection feature using the Dolos library.

### Prerequisites

To use this feature, you'll need to set up the following:

1. **Node.js**: Ensure you have [Node.js](https://nodejs.org/) installed on your
   system. We recommend installing it via [nvm](https://github.com/nvm-sh/nvm)

2. **Dolos Library**: Navigate to the plagiarism detector directory and install
   the NodeJS requirements:

   ```console
   cd extras/plagiarism_detector
   npm install
   ```

3. **Popular Kaggle competition Kernels**: We compare each competition
   submission to the most popular kernels for that competition. To do this, of
   course, we need the kernels. We provide a script for downloading these
   kernels:

   ```console
   python extras/kernels/run.py download-kernels -c <competition-id>
   ```

   This will download the top kernels for the requested competition into its
   folder in `mlebench/competitions/`, allowing us to run the plagiarism
   detector against these kernels.

   The top kernels have been pre-determined by us. If you wish, you can
   re-determine the top kernels by running the following command:

   ```console
   python extras/kernels/run.py kernel-refs <competition-id>
   ```

   This will update the `kernels.txt` file in the competition's folder with the
   ids of most recent popular kernels for that comp. You can then re-download
   the kernels with the `download-kernels` command from above.

### Usage

Once you have these prerequisites setup, you can use the plagiarism detection
feature with the following command:

```console
python plagiarism_detector/run.py --submission <path-to-submission-jsonl> --output-dir <output-dir>
```

For each submission in the provided submission JSONL file, this will check for
similarity between the submitted code and the competition's associated kernels.
A separate JSONL report for each submission will be saved to the output
directory, containing the similarity scores between the submitted code and each
kernel.
