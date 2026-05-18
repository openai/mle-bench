## Description

### Build a model to transcribe ancient Kuzushiji into contemporary Japanese characters

Imagine the history contained in a thousand years of books. What stories are in those books? What knowledge can we learn from the world before our time? What was the weather like 500 years ago? What happened when Mt. Fuji erupted? How can one fold 100 cranes using only one piece of paper? The answers to these questions are in those books.

Japan has millions of books and over a billion historical documents such as personal letters or diaries preserved nationwide. Most of them cannot be read by the majority of Japanese people living today because they were written in "Kuzushiji".

Even though Kuzushiji, a cursive writing style, had been used in Japan for over a thousand years, there are very few fluent readers of Kuzushiji today (only 0.01% of modern Japanese natives). Due to the lack of available human resources, there has been a great deal of interest in using Machine Learning to automatically recognize these historical texts and transcribe them into modern Japanese characters. Nevertheless, several challenges in Kuzushiji recognition have made the performance of existing systems extremely poor. (More information in [About Kuzushiji](https://www.kaggle.com/c/kuzushiji-recognition/overview/about-kuzushiji))

This is where you come in. The hosts need help from machine learning experts to transcribe Kuzushiji into contemporary Japanese characters. With your help, Center for Open Data in the Humanities (CODH) will be able to develop better algorithms for Kuzushiji recognition. The model is not only a great contribution to the machine learning community, but also a great help for making millions of documents more accessible and leading to new discoveries in Japanese history and culture.

![umgy001-010-smallannomasked](http://static.mxbi.net/umgy001-010-smallannomasked.jpg)

### Hosts

[Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/) conducts research and development to enhance access to humanities data using state-of-the-art technology in informatics and statistics.

[The National Institute of Japanese Literature (NIJL)](https://www.nijl.ac.jp/en/) is an institution which strives to serve researchers in the field of Japanese literature as well as those working in various other humanities, by collecting in one location a vast storage of materials related to Japanese literature gathered from all corners of the country.

[The National Institute of Informatics (NII)](https://www.nii.ac.jp/en/) is Japan's only general academic research institution seeking to create future value in the new discipline of informatics. NII seeks to advance integrated research and development activities in information-related fields, including networking, software, and content.

#### Official Collaborators
Mikel Bober-Irizar ([anokas](https://www.kaggle.com/anokas)) Kaggle Grandmaster and Alex Lamb (MILA. Quebec Artificial Intelligence Institute)

## Evaluation

Submissions are evaluated on a modified version of the [F1 Score](https://en.wikipedia.org/wiki/F1_score). To score a true positive, you must provide center point coordinates that are within the ground truth bounding box and a matching label. The ground truth bounding boxes are defined in the format `{label X Y Width Height}`, so if the ground truth label is `U+003F 1 1 10 10` then a prediction of `U+003F 3 3` would pass. You can find [a Python version of the metric here](https://gist.github.com/SohierDane/a90ef46d79808fe3afc70c80bae45972).

### Submission File

For each image in the test set, you must locate and identify all of the kuzushiji characters. Not every image will necessarily be scored. The file should contain a header and have the following format:

```
image_id,labels
image_id,{label X Y} {...}
```

Do not make more than 1,200 predictions per page.

### Timeline

-   **October 7, 2019 - Entry deadline**. You must accept the competition rules before this date in order to compete.
-   **October 7, 2019 - Team Merger deadline**. This is the last day participants may join or merge teams.
-   **October 14, 2019 - Final submission deadline**.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

5 qualifying teams will each receive a $3,000 research grant as follows. In order to be eligible, teams should complete an interest form, and will have to provide documentation and open-source their solutions. Further details, along with the interest form, will be posted in the discussion forum upon the competition close.

-   First Prize: $3,000

-   Second Prize: $3,000

-   Third Prize: $3,000

-   Fourth Prize: $3,000

-   Fifth Prize: $3,000

Eligible teams will be invited to present and receive awards at "Japanese Culture and AI" symposium in Tokyo, Japan on November 11, 2019. Attendees presenting in person are responsible for all costs associated with travel expenses. Attending the symposium is not required to participate in the competition.

## About Kuzushiji

Kuzushiji, a cursive writing style, was used in Japan for over a thousand years, beginning in the 8th century. Over 3 million books, on a diverse array of topics such as literature, science, mathematics and cooking are preserved today. However, the standardization of Japanese textbooks known as the "Elementary School Order" in 1900, removed Kuzushiji from regular school curriculum, as modern japanese print became popular. As a result, most Japanese natives today cannot read books written or printed just 120 years ago.

Since Chinese characters entered Japan in the 8th century, the Japanese language has been written using Kanji (Chinese characters in the Japanese language) in official records. However, from the late 9th century, the Japanese began to add their own character sets: Hiragana and Katakana, which derive from different ways of simplifying Kanji. Individual Hiragana and Katakana characters don't contain independent semantic meaning, but instead carry phonetic information (like letters in the English alphabet).

In Kuzushiji documents, Kanji, Hiragana and Katakana are all used. However, the number of character types in each document varies by genre. For example, story books are mostly written in Hiragana while formal records are written mainly in Kanji.

### Challenges in Kuzushiji recognition

**Large number of character types**: The total number of unique characters in the Kuzushiji dataset is over 4300. However, the frequency distribution is very long-tailed and a large fraction of the characters (Kanji with very specific meaning) may only appear once or twice in a book. Therefore, the dataset is highly unbalanced.

**Hentaigana**: One characteristic of Classical Hiragana or Hentaigana ("character variations'') is that many characters which can only be written a single way in modern Japanese can be written in many different ways in Kuzushiji.\
For example, this image shows that Hiragana *Ha* (は) can be written in a few different ways.\
![](http://codh.rois.ac.jp/competition/kaggle/images/ha.jpg)

**Similarity between characters**: A few characters in Kuzushiji look very similar and it is hard to tell what character it is without considering the above character as context.\
For example, in the image below the red circles show 3 types of characters: *Ku* (く), an iteration mark and *Te* (て).\
![](http://codh.rois.ac.jp/competition/kaggle/images/ku.jpg)

**Connectedness and overlap between characters**: Kuzushiji was written in a cursive script, and hence in many cases, characters are connected or overlap which can make the recognition task difficult.\
In the image below, the bounding boxes in the image below show that characters overlap. The color of the boxes are for visualization and don't contain any specific meaning.\
![](http://codh.rois.ac.jp/competition/kaggle/images/overlapped.jpg)

**Various layouts**: The layout of Kuzushiji characters (while normally arranged into columns) does not follow a single simple rule, so it is not always trivial to express the characters as a sequence. Some examples of this include characters being written to wrap around or even integrate into illustrations. Sometimes, characters are written in a diagonal line.\
![](http://codh.rois.ac.jp/competition/kaggle/images/layout.jpg)

## Citation

anokas, Asanobu KITAMOTO, Elizabeth Park, Sohier Dane, TheNuttyNetter, tkasasagi, Wendy Kan. (2019). Kuzushiji Recognition. Kaggle. https://kaggle.com/competitions/kuzushiji-recognition


## Dataset Description

Kuzushiji is a cursive script that was used in Japan for over a thousand years that has fallen out of use in modern times.

Vast portions of Japanese historical documents now cannot be read by most Japanese people. By helping to automate the transcription of kuzushiji you will contribute to unlocking a priceless trove of books and records.

The specific task is to locate and classify each kuzushiji character on a page. While complete bounding boxes are provided for the training set, only a single point within the ground truth bounding box is needed for submissions.

There are a few important things to keep in mind for your submissions:

- Some images only contain illustrations, in which case no labels should be submitted for the page.

- Kuzushiji text is written such that annotations are placed between the columns of the main text, usually in a slightly smaller font. See [the characters with no bounding boxes in this image](http://codh.rois.ac.jp/competition/kaggle/images/rubi.jpg) for examples. The annotation characters should be ignored; labels for them will be scored as false positives.

- You can occasionally see through especially thin paper and read characters from the opposite side of the page. Those characters should also be ignored.

### Files
-----

- **train.csv** - the training set labels and bounding boxes.
    - `image_id`: the id code for the image.
    - `labels`: a string of all labels for the image. The string should be read as space separated series of values where `Unicode character`, `X`, `Y`, `Width`, and `Height` are repeated as many times as necessary.
- **sample_submission.csv** - a sample submission file in the correct format.
    - `image_id`: the id code for the image
    - `labels`: a string of all labels for the image. The string should be read as space separated series of values where\
        `Unicode character`, `X`, and `Y` are repeated as many times as necessary. The default label predicts that there are the same two characters on every page, centered at pixels `(1,1)` and `(2,2)`.
- **unicode_translation.csv** - supplemental file mapping between unicode IDs and Japanese characters.
- **[train/test]_images.zip** - the images.