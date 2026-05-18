# Familiarity experiment

We measure the familiarity of gpt-4o with each competition's description and top discussion posts. We compare this familiarity score with the benchmark score of each competition.

This particular experiment uses separate requirements since it requires an older version of the OpenAI API to support base models. We also include some additional packages that are not necessary for the main library.

Install:
```bash
pip install -r requirements.txt
playwright install  # Install playwright for scraping discussion posts
```

## Getting the top discussion posts

We use the Meta Kaggle dataset (`kaggle datasets download -d kaggle/meta-kaggle`) with https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions to get the links for top discussion posts for each competition. For reproducibility, we record the top-5 discussion posts for each competition in the `comps_to_urls.json` file, which is what we use in our experiments.

Given the list of URLs, we use playwright to scrape the text for all the discussion posts.
```bash
python get_discussion_posts.py
```

## Getting the familiarity scores

Given a set of documents (we use the competition description and top-5 discussion posts for each comp), we get the gpt-4o base model's log-likelihood for each token in each document and calculate the mean probability over all tokens (in practice, we sample up to `--n_samples` tokens, which defaults to 1000). We treat the mean probability as a proxy of model's "familiarity" with a given competition. We plot this familiarity measure against the benchmark score (performance) for each competition (for which we assume the relevant grading reports are available at `../../runs/`).
```bash
python familiarity.py
```

Calculating the mean probability over all tokens takes a while (an hour or two), so we save the results in `familiarity_and_dates.json`. You can skip re-calculating this if you just want to run the plots:
```bash
python familiarity.py --existing-results familiarity_and_dates.json
```