# Task

Categorise plant images.

# Metric

Macro F1 score. In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged. 

# Submission Format 

For each image `Id`, you should predict the corresponding image label (`category_id`) in the `Predicted` column. The submission file should have the following format:
```
Id,Predicted
0,1
1,27
2,42
...
```

# Dataset

In addition to the images, we also include a hierarchical taxonomic structure of `category_id`. The `categories` in the `training_metadata.json` contain three levels of hierarchical structure, `family` - `genus` - `species`, from the highest rank to the lowest rank. One can think about this as a directed graph, where families are the root nodes and the species are the leaf nodes. Please note that `species` is only unique under its parent node `genus`, which means that we can find multiple categories with the same `species` name under different `genus` names. This is due to the taxonomic nature that plants are named after, and the `genus`-`species` pair is always unique in our data.

Pairwise phylogenetic distances among genera is included in key name `distances` from `train_metadata.json`.

The training set images are organized in subfolders `h22-train/images/<subfolder1>/<subfolder2>/<image_id>.jpg`, where `<subfolder1>` and `<subfolder2>` comes from the first three and the last two digits of the image_id. Image_id is a result of combination between `<category_id>` and unique numbers that differentiates images within plant taxa. Please be mindful that `category_id`s are unique, but not complete. Taxa are originally numbered from 1 to 15505, but the competition data has 15501 taxa because we lost four taxa during data cleaning process.

The test set images are organized in subfolders `test/images/<subfolder>/<image id>.jpg`, where `<subfolder>` corresponds to the integer division of the `image_id` by 1000. For example, a test image with and `image_id` of `8005`, can be found at `h22-test/images/008/test-008005.jpg`