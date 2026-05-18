Initial models

Shortly after the competition started, @prikshitsingla discovered that the number of NaNs in a row is a good feature for predicting the target—and indeed, it has been shown to be correlated with the target value (post). This allowed me to elevate my AUC score above a solid 0.8. Amazing. A good understanding of the data is indeed crucial for building models on it. I haven’t done much EDA in this competition, and I hope I can improve in that area in the future.

Then, as always, I began experimenting with XGBoost and LightGBM classifiers and used Optuna to tune hyperparameters. Due to the large size of the data file, I used random sampling to work on 20% of the entire dataset. This enabled me to run Optuna quickly—with 5-fold cross-validation—and rapidly narrow down a combination of hyperparameters that performed quite well. Unfortunately, further Optuna searches led nowhere and yielded no improvement on the public LB score. Overall, I was not particularly successful in uncovering many high-performing models—so I began referring to others’ work and adopted stacking.

Stacking

Then came the stacking phase. The principle is well explained by @abhishek22211 in his post. Essentially, one must perform cross-validation on several so-called level-0 models, then use their out-of-fold predictions to train a level-1 meta-model for the final prediction.

Thanks to @vishwas21 (notebook), @manabendrarout (notebook), and @mlanhenke (notebook), I was able to gather a total of 34 models at my L0 level.

In terms of stacking, what worked really well were:

- Using LinearRegression instead of logistic regression as the L1 meta-model.  
- Reusing the outputs of both the L1 and L0 models as inputs to an L2 meta-model—which again is a LinearRegression.  
- Performing some target encoding based on the number of NaNs in a row, and using various random seeds to replicate certain models, which appeared effective. Also, thanks to @edrickkesuma for the excellent post on power blending—I learned some new techniques!

What did *not* help my stacking score:

- Using XGBoost or LightGBM classifiers as the meta-model, and tuning them with Optuna at the L1 level.  
- Blindly adding mediocre-performing models (e.g., RF, LogisticRegression) at the L0 level solely for diversity’s sake.  
- Pseudo-labeling (post).  

After this, I was about halfway up my climb toward my final ranking—around 30th place on the public LB.

“One model voting” and “Fit to all”

Then, inspired by @martynovandrey’s post, I began replicating LightGBM models from  
@hiro5299834 (notebook), @ivankontic (notebook), @mlanhenke (notebook), and @realtimshady (notebook), plus a custom XGBoost model of my own—each trained with 7 different `random_state` seeds and blended—yielding a noticeable score improvement (although in hindsight, the resulting rise on the public LB is more pronounced than the rise on the private LB. So the “effectiveness” of these improvements may be less than, say, what stacking delivers).

A further improvement was achieved by fitting a model to *all* training data and increasing the `n_estimators` value beyond its “optimal” value found via cross-validation with early stopping. The rationale here is:

- By training a “fit-to-all” model on the full dataset, it should be possible to push its complexity further to learn better—hopefully without falling into overfitting.  
- And sometimes early stopping may have been triggered too early during CV.  

This may be the most important lesson I learned in this competition. In fact, I had grown quite accustomed to relying entirely on cross-validation for test-data prediction—that is:

- Use KFold or StratifiedKFold to split the entire training set into K folds, with `early_stopping_rounds` around 300–500.  
- At each CV iteration, train a new model on the non-validation folds, evaluate its score on the validation fold, and use this model to predict on the test data.  
- Blend all test predictions across CV iterations to produce the final prediction.  

Although there is some benefit in using a blended prediction from all CV iterations, one should also consider that *no single model* has ever seen or been trained on the *entire* training set—a potential missed opportunity. So the bottom line is: trust the hyperparameters validated via cross-validation, but later verify the `n_estimators` value by training on the full dataset to see whether any improvement can be achieved on the public LB.

Finally, I blended my L1 prediction from my stacking notebook, “fit-to-all” predictions from 5 models, and the impressive submission from @mlanhenke (notebook) for my final submission. To compare all predictions, the density plot of their predicted values looks like this:

There are indeed some models that produced more eccentric results than others. I considered applying different weights or power blending here—but decided against it. In the end, this simple approach worked rather well and placed me among the top 10 on the private LB.