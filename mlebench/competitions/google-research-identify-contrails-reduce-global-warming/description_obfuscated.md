# Task

Identify contrails in satellite imagery.

# Metric

Global Dice coefficient. The Dice coefficient formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where X is the entire set of predicted contrail pixels for all observations in the test data and Y is the ground truth set of all contrail pixels in the test data.

# Submission Format

Use run-length encoding. For example, '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

Use a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

Empty predictions must be marked with '-' in the submission file.

The file should contain a header and have the following format:

```
record_id,encoded_pixels  
1000834164244036115,1 1 5 1  
1002653297254493116,-  
etc.
```

# Dataset 

Some key labeling guidance:
- Contrails must contain at least 10 pixels
- At some time in their life, Contrails must be at least 3x longer than they are wide
- Contrails must either appear suddenly or enter from the sides of the image
- Contrails should be visible in at least two image

A sequence of images at 10-minute intervals are provided. Each example (`record_id`) contains exactly one labeled frame.

- **train/** - the training set; each folder represents a `record_id` and contains the following data:
    - **band_{08-16}.npy**: array with size of `H x W x T`, where `T = n_times_before + n_times_after + 1`, representing the number of images in the sequence. There are `n_times_before` and `n_times_after` images before and after the labeled frame respectively. In our dataset all examples have `n_times_before=4` and `n_times_after=3`. Each band represents an infrared channel at different wavelengths and is converted to brightness temperatures based on the calibration parameters. The number in the filename corresponds to the GOES-16 ABI band number. Details of the ABI bands can be found [here](https://www.goes-r.gov/mission/ABI-bands-quick-info.html).
    - **human_individual_masks.npy**: array with size of `H x W x 1 x R`. Each example is labeled by `R` individual human labelers. `R` is not the same for all samples. The labeled masks have value either 0 or 1 and correspond to the `(n_times_before+1)`-th image in `band_{08-16}.npy`. They are available only in the training set.
    - **human_pixel_masks.npy**: array with size of `H x W x 1` containing the binary ground truth. A pixel is regarded as contrail pixel in evaluation if it is labeled as contrail by more than half of the labelers.
- **validation/** - the same as the training set, without the individual label annotations; it is permitted to use this as training data if desired
- **test/** - the test set; your objective is to identify contrails found in these records.
- **{train|validation}_metadata.json** - metadata information for each record; contains the timestamps and the projection parameters to reproduce the satellite images.
- **sample_submission.csv** - a sample submission file in the correct format
