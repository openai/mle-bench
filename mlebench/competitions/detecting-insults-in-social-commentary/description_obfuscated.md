# Task

Detect if a comment is insulting.

# Metric

Area under the Receiver Operating Curve (AUC).

# Submission Format

Your predictions should be a number in the range [0,1].

See 'sample_submissions_null.csv" for the correct format.

# Dataset

The label is either 0 meaning a **neutral** comment, or 1 meaning an **insulting** comment.

The first attribute is the time at which the comment was made. It is sometimes blank, meaning an accurate timestamp is not possible. It is in the form "YYYYMMDDhhmmss" and then the Z character.

The second attribute is the unicode-escaped text of the content, surrounded by double-quotes. The content is mostly english language comments, with some occasional formatting.

## Guidelines

- We are looking for comments that are intended to be insulting to a person who is a part of the larger blog/forum conversation. 
- We are NOT looking for insults directed to non-participants (such as celebrities, public figures etc.). 
- Insults could contain profanity, racial slurs, or other offensive language. But often times, they do not. 
- Comments which contain profanity or racial slurs, but are not necessarily insulting to another person are considered not insulting.
- The insulting nature of the comment should be obvious, and not subtle. 

There may be a small amount of noise in the labels as they have not been meticulously cleaned. However, contenstants can be confident the error in the training and testing data is `< 1%`.