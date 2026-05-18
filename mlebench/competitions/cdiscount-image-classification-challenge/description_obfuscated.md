# Task

Predict the category of a product based on its image.

# Metric

Categorization accuracy.

# Submission Format

```
_id,category_id
2,1000000055
5,1000016018
6,1000016055
etc.
```

# Dataset

- **train.bson** - (Size: 58.2 GB) Contains a list of 7,069,896 dictionaries, one per product. Each dictionary contains a product id (key: _id), the category id of the product (key: category_id), and between 1-4 images, stored in a list (key: imgs). Each image list contains a single dictionary per image, which uses the format: {'picture': b'…binary string…'}. The binary string corresponds to a binary representation of the image in JPEG format.
- **train_example.bson** - Contains the first 100 records of train.bson.
- **test.bson** - (Size: 14.5 GB) Contains a list of 1,768,182 products in the same format as train.bson, except there is no category_id included.
- **category_names.csv** - Shows the hierarchy of product classification. Each category_id has a corresponding level1, level2, and level3 name, in French. The category_id corresponds to the category tree down to its lowest level.
- **sample_submission.csv** - A sample submission file in the correct format.