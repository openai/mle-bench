# Operand: All MLE-Benchmark Runs
Date Submitted: 10-06-2025

All Operand runs (including logs, generated code, and grading reports) for every seed can be found in this public GitHub repository: https://github.com/ramgorthi04/OperandLinear-MLE-Bench

The README.md file from that repository is included below for reference.

## Directory Tree
```bash
└── MLE_Submission
    ├── 0_Environment_Setup
    ├── 1_Verification_Scripts
    ├── 2_Architecture
    ├── aerial-cactus-identification
    ├── alaska2-image-steganalysis
    ├── aptos2019-blindness-detection
    ├── cassava-leaf-disease-classification
    ├── cdiscount-image-classification-challenge
    ├── chaii-hindi-and-tamil-question-answering
    ├── champs-scalar-coupling
    ├── denoising-dirty-documents
    ├── detecting-insults-in-social-commentary
    ├── dog-breed-identification
    ├── dogs-vs-cats-redux-kernels-edition
    ├── facebook-recruiting-iii-keyword-extraction
    ├── freesound-audio-tagging-2019
    ├── google-quest-challenge
    ├── h-and-m-personalized-fashion-recommendations
    ├── herbarium-2020-fgvc7
    ├── herbarium-2021-fgvc8
    ├── herbarium-2022-fgvc9
    ├── histopathologic-cancer-detection
    ├── hotel-id-2021-fgvc8
    ├── icecube-neutrinos-in-deep-ice
    ├── imet-2020-fgvc7
    ├── inaturalist-2019-fgvc6
    ├── iwildcam-2019-fgvc6
    ├── iwildcam-2020-fgvc7
    ├── jigsaw-toxic-comment-classification-challenge
    ├── kuzushiji-recognition
    ├── leaf-classification
    ├── learning-agency-lab-automated-essay-scoring-2
    ├── lmsys-chatbot-arena
    ├── mlsp-2013-birds
    ├── multi-modal-gesture-recognition
    ├── new-york-city-taxi-fare-prediction
    ├── nfl-player-contact-detection
    ├── nomad2018-predict-transparent-conductors
    ├── osic-pulmonary-fibrosis-progression
    ├── petfinder-pawpularity-score
    ├── plant-pathology-2020-fgvc7
    ├── plant-pathology-2021-fgvc8
    ├── predict-volcanic-eruptions-ingv-oe
    ├── random-acts-of-pizza
    ├── ranzcr-clip-catheter-line-classification
    ├── rsna-2022-cervical-spine-fracture-detection
    ├── rsna-breast-cancer-detection
    ├── seti-breakthrough-listen
    ├── siim-covid19-detection
    ├── siim-isic-melanoma-classification
    ├── smartphone-decimeter-2022
    ├── spooky-author-identification
    ├── stanford-covid-vaccine
    ├── tabular-playground-series-dec-2021
    ├── tabular-playground-series-may-2022
    ├── tensorflow-speech-recognition-challenge
    ├── text-normalization-challenge-english-language
    ├── text-normalization-challenge-russian-language
    ├── tgs-salt-identification-challenge
    ├── the-icml-2013-whale-challenge-right-whale-redux
    ├── tweet-sentiment-extraction
    ├── us-patent-phrase-to-phrase-matching
    ├── uw-madison-gi-tract-image-segmentation
    ├── ventilator-pressure-prediction
    ├── vesuvius-challenge-ink-detection
    ├── vinbigdata-chest-xray-abnormalities-detection
    └── whale-categorization-playground
```

### Verification Scripts (1_Verification_Scripts)
`calc_medal_rates.py` produces `medal_rates.csv`, which reports mean and SEM by parsing all competition_results.json files across all problems and seeds.

### Environment Setup (0_Environment_Setup)
Base and agent Dockerfiles, docker compose, agent environment requirements, and orchestrator requirements are provided.

### Architecture Specifications (2_Architecture)
PDF covering key architecture specifications, MLE Benchmark compliance.

### Logs
`full_history.json` contains the turn-by-turn log of the Operand Quant. All .ipynb files have been persisted and included for reproducibility.

## Notes
Due to infrastructure limitations and compute costs, we were unable to evaluate the agent on a number of problems in the hard subset. Nevertheless, for fairness, our reported scores are calculated across the full set of 75 problems. As we had already achieved state-of-the-art results without these few, we did not further optimize our infrastructure to support the remaining evaluations.

## SEM Calculation
Following leaderboard convention, the reported “Any Medal (%) ± SEM” values use a binomial standard error model that treats each seed run as an independent Bernoulli trial. This corresponds to the standard error of a binomial proportion over all seeds (225 total for the full benchmark).