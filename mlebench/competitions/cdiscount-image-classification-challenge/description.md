# Overview

## Description

![Cdiscount](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3333/media/Cdiscount.png)

[Rules Update:](https://www.kaggle.com/c/cdiscount-image-classification-challenge/rules) The CDiscount team has updated their rules to allow for use of this dataset for research and academic purposes only. To access the data, go to rules and accept the terms to download the data.

[Cdiscount.com](https://www.cdiscount.com/) generated nearly 3 billion euros last year, making it France’s largest non-food e-commerce company. While the company already sells everything from TVs to trampolines, the list of products is still rapidly growing. By the end of this year, Cdiscount.com will have over 30 million products up for sale. This is up from 10 million products only 2 years ago. Ensuring that so many products are well classified is a challenging task.

Currently, Cdiscount.com applies machine learning algorithms to the text description of the products in order to automatically predict their category. As these methods now seem close to their maximum potential, Cdiscount.com believes that the next quantitative improvement will be driven by the application of data science techniques to images.

In this challenge you will be building a model that automatically classifies the products based on their images. As a quick tour of Cdiscount.com's website can confirm, one product can have one or several images. The data set Cdiscount.com is making available is unique and characterized by superlative numbers in several ways:

- Almost 9 million products: half of the current catalogue
- More than 15 million images at 180x180 resolution
- More than 5000 categories: yes this is quite an extreme multi-class classification!

## Evaluation

### Goal

The goal of this competition is to predict the category of a product based on its image(s). Note that a product can have one or several images associated. For every product*_id* in the test set, you should predict the correct *category_id*.

### Metric

This competition is evaluated on the categorization accuracy of your predictions (the percentage of products you get correct).

### Submission File

For each *_id* in the test set, you must predict a *category_id*. The file should contain a header and have the following format:

```
_id,category_id
2,1000000055
5,1000016018
6,1000016055
etc.
```

## Prizes

- 1st place - $20,000
- 2nd place - $10,000
- 3rd place - $5,000

## Timeline

- **December 7, 2017** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **December 7, 2017** - Team Merger deadline. This is the last day participants may join or merge teams.
- **December 14, 2017** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About Cdiscount

![Cdiscount](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3333/media/Cdiscount.png)

[Cdiscount.com](https://www.cdiscount.com/), a subsidiary company of the Casino Group, is a major online retailer in France. The company has reported outstanding growth and generated a business volume of nearly 3 billion euros in 2016 supported by a constantly growing Marketplace, with products sold by more than 9,000 partnering merchants.

Focusing on four core values: proximity, audacity, engagement and reliability, Cdiscount.com strives to make everyday products and services affordable to everyone, while working hard to better understand and serve the customers’ needs.

One way to reach these goals is to develop the offer. Cdiscount.com's product catalog is growing rapidly and should reach more than 30 million items by the end of the year. In order to support this development and allow clients to browse intuitively through such a large range of products, it is important that each item is well-classified.

Therefore, Cdiscount.com believes that data science is the way forward to automatically select the best category for each new product and improve the shopping experience.

### Key figures

- 3 billion euros GMV
- 8 million active customers
- 9,000 partnering merchants
- 1,500 employees

Don't hesitate to have a look at Cdiscount.com's [job offers](https://emploi.cdiscount.com/). Data scientists are being hired!

## Citation

CdiscountDS, inversion, Jeremy B, Mark McDonald. (2017). Cdiscount’s Image Classification Challenge. Kaggle. https://kaggle.com/competitions/cdiscount-image-classification-challenge

# Data

## Dataset Description

### Update:

At the request of the sponsor, the data has been removed post-competition.

### BSON Files

[BSON](https://en.wikipedia.org/wiki/BSON), short for Bin­ary JSON, is a bin­ary-en­coded seri­al­iz­a­tion of JSON-like doc­u­ments, used with [MongoDB](https://en.wikipedia.org/wiki/MongoDB). [This kernel](https://www.kaggle.com/inversion/processing-bson-files) shows how to read and process the BSON files for this competition.

### File Descriptions

**Please Note:** The train and test files are very large!

- **train.bson** - (Size: 58.2 GB) Contains a list of 7,069,896 dictionaries, one per product. Each dictionary contains a product id (key: _id), the category id of the product (key: category_id), and between 1-4 images, stored in a list (key: imgs). Each image list contains a single dictionary per image, which uses the format: {'picture': b'…binary string…'}. The binary string corresponds to a binary representation of the image in JPEG format. [This kernel](https://www.kaggle.com/inversion/processing-bson-files) provides an example of how to process the data.
- **train_example.bson** - Contains the first 100 records of train.bson so you can start exploring the data before downloading the entire set.
- **test.bson** - (Size: 14.5 GB) Contains a list of 1,768,182 products in the same format as train.bson, except there is no category_id included. The objective of the competition is to predict the correct category_id from the picture(s) of each product id (_id). The category_ids that are present in Private Test split are also all present in the Public Test split.
- **category_names.csv** - Shows the hierarchy of product classification. Each category_id has a corresponding level1, level2, and level3 name, in French. The category_id corresponds to the category tree down to its lowest level. This hierarchical data may be useful, but it is not necessary for building models and making predictions. All the absolutely necessary information is found in train.bson.
- **sample_submission.csv** - Shows the correct format for submission. It is highly recommended that you zip your submission file before uploading for scoring.