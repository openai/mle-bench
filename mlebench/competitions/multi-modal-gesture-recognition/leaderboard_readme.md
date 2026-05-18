Note: The leaderboard.csv has been manually modified and differs from the raw leaderboard
downloadable from https://www.kaggle.com/competitions/multi-modal-gesture-recognition/leaderboard.

In the original leaderboard, the bottom 35 submissions curiously report scores of `999999.00000`.
This includes a "Random predictions" team, suggesting that these scores correspond to a random
baseline similar to the `randomPredictions.csv` available in the raw dataset.

However, this score is significantly higher than what we'd expect from a straightforward
implementation of the described metric, which would have a random baseline scoring close to but
less than `1.0`. Indeed, the random baseline in our sample submission scores `0.91966` on our new
test set using our grader implementation, which is consistent with our expectations.

Ranking our sample submission against the original leaderboard would achieve a bronze medal, as it
beats all the submissions scoring `999999.00000`. This would be a misleading medal achievement
because the score is in fact quite weak.

To avoid this misleading result, we have decided to remove all the `999999.00000` scores from the
leaderboard.

Here are the medal thresholds based on the original leaderboard:
```
    "gold_threshold": 0.17872,
    "silver_threshold": 0.322,
    "bronze_threshold": 999999.0,
    "median_threshold": 999999.0,
```

And here are the medal thresholds based on our new modified leaderboard, showing more difficult
but reasonable thresholds for bronze and median:
```
    "gold_threshold": 0.13304,
    "silver_threshold": 0.1769,
    "bronze_threshold": 0.24926,
    "median_threshold": 0.322,
```
