# Overview

## Overview

### Prizes

#### How Does the Prize Work?

The \$10,000 prize pool will be split between the top two finishers for the classification challenge and the top visualization submission as follows:

**First place**: \$7,000

**Second place**:  \$2,500

**Visualization Prospect**: \$500

But wait, there's more...

**Recruiting Competition**

Do you want a chance to build a crucial component of Web 2.0 infrastructure: the defense against spam and abuse? Do you live and breathe machine learning and data-mining?\
We're looking for someone whose passion lies in the invention and application of cutting-edge machine learning and data-mining techniques. Impermium is an engineering-driven startup dedicated to helping consumer web sites protect themselves from "social spam," account hacking, bot attacks, and more.

Impermium will review the top entries and offer interviews to the creators of those submissions which are exceptional.

**Please note: You must compete as an individual in recruiting competitions.  You may only use the data provided to make your predictions.  Impermium will review the code of the top participants before deciding whether to offer an interview.**

**Due to visa issuance delays, Impermium is not able to sponsor new H1B applicants but are happy to support H1B transfers, permanent residents and US citizens.**

### Description

#### [Visualization Track now open >>](https://www.kaggle.com/c/detecting-insults-in-social-commentary/prospector)

#### Do you think you can take on the crudest, meanest trolls on the internet? Okay then, game on!

The challenge is to detect when a comment from a conversation would be considered insulting to another participant in the conversation. Samples could be drawn from conversation streams like news commenting sites, magazine comments, message boards, blogs, text messages, etc.

The idea is to create a generalizable single-class classifier which could operate in a near real-time mode, scrubbing the filth of the internet away in one pass.

#### _Prizes you say?_

Besides the prize money, eternal fame and glory, monuments in your honor, and the admiration of friend and foe alike, you get a chance for an interview at Impermium for the Principal Data Engineer role.

In addition, there will be a Visualization prospect attached to the contest.  Slice and dice the data to show us the most amazing, informative and thought-provoking infographics, diagrams or plots! Submission open 1 week before the contest ends. 

*(Please note that referenced fame, glory, monuments and admiration are to be bestowed only in the imagination of the contestants)*

### About The Sponsor

#### What are social spam and abuse?

Impermium defines social spam as any unwanted content that someone experiences on social network or in user-generated content sites on the Internet. Social web abuse involves malicious efforts to take advantage of social channels, including password hacking, malware placement, fake follower and friending schemes, account takeovers, bogus registrations, and more.

Five years ago, social spam and abuse practically didn't exist. But this year, 90% of social network users have experienced it -- and victims are beginning to lose hundreds of thousands of dollars. Because social networks are relatively new to users, and cybercriminals are already inside them, we can expect the damages to mount for many years to come.

#### Who is Impermium?

Impermium is an engineering-driven startup dedicated to helping consumer web sites protect themselves from "social spam," account hacking, bot attacks, and more. We are funded by some of the most experienced and influential investors in the Valley, and already receiving strong enthusiasm and rapid uptake.

Impermium provides a subscription-based service that protects against the biggest threats today, allowing social application developers to remain focused on their core business. Built by the former leaders of the Yahoo! anti-spam and security teams, the company's RESTful Web Services-based solution enables site operators to prevent and mitigate most types of suspicious and abusive transactions, with real-time and batch monitoring of Web transactions, and hosted mitigation solutions to manage suspicious users.

Backed by some of the most prominent technology investors in Silicon Valley including Accel Partners, Charles River Ventures, Greylock Partners, Highland Capital Partners, and the Social+Capital Partnership, Impermium provides the essential anti-abuse and security services for many of the Internet's most popular consumer web sites and social media engagement platforms.

### Job Description

#### Principal Data Engineer at Impermium in Redwood City, CA

**As Impermium continues to build out its Internet-scale user-generated content classification system, we're looking for someone whose passion lies in the invention and application of cutting-edge machine learning and data-mining techniques. Our system already classifies tens of millions of "social transactions" every day, looking for spam, abuse, fraud, and other bad behavior -- as Data Engineer, you'll join our team to help guide and shape our algorithm and classifier development. Read on if the following excites you:**

- Locality Sensitive Hashing!
- Random Forest Ensemble Classifiers!
- Stochastic Gradient Boosting Distributed Decision Trees!

**Most large-scale abuse classification systems break down due to non-I.I.D. document distributions, an over-reliance on exhaustive "ground truth" training corpora, and an adversary who continually adapts to specific weaknesses in the classifier.  The Impermium Data Engineer will help in our pioneering work to overcome these historical limitations.\
The ideal candidate is a highly knowledgeable, all-star computer engineer, with a strong background in machine learning, data mining, and distributed computing. This candidate must have previous, hands-on experience turning conversations into prototypes and prototypes into products -- ideally in a startup environment.**

**You'll fit right in if:**

- You are a self-managed, high-energy individual
- You possess exceptional communication skills with the ability to clearly articulate your engineering and product ideas with both team-members and customers
- You are absolutely confident in your ability to design scalable, principled, practical classification and clustering algorithms that operate within the near-real-time constraints of the abuse domain

**Requirements:**

- 5+ years experience creating prototypes that are shipped to market (production systems)
- 5+ years experience in software product development - with the core focus being on mathematical and / or statistical algorithms
- Well-versed in a modern general purpose programming language such as Java/C++/Scala/Python
- Well-versed with unix command line tools and one or more scripting language such as awk/perl/shell
- Proven ability to develop and execute sophisticated data mining & modeling
- Experience working with "Big Data" systems and platforms
- NLP (natural language processing) experience is a plus
- Publications and/or patents are a plus
- MS/PH.D in Computer Science recommended

**Impermium offers you:**

- A chance to build a crucial component of Web 2.0 infrastructure: the defense against spam and abuse
- A dynamic, technology-driven work environment in our brand new office, convenient to 101 and Caltrain
- A highly influential and visible role with direct impact on foundational product and engineering direction
- The opportunity to work alongside a highly talented, experienced founding team

Due to visa issuance delays, Impermium is not able to sponsor new H1B applicants but are happy to support H1B transfers, permanent residents and US citizens.

### Evaluation

This is a single-class classification problems.  Your predictions should be a number in the range [0,1]  where 0 indicates 0% probability of comment being an insult, and 1 represents 100% insult.

 All predictions should be in the first column of your submission file. Please see 'sample_submissions_null.csv" for the correct format.

The evaluation metric for this competition is the Area under the Receiver Operating Curve (AUC).   This evaluation metric penalizes wrong predictions made with high probability.  For more about this metric, check out the [AUC](https://www.kaggle.com/wiki/AreaUnderCurve) page on the Kaggle Wiki.

**Please note, winners for the competition will be determined by their performance on an additional verification set that will be released at the end of the competition.  See Timeline tab for details.**

### Timeline

Tuesday, August 7th: Competition launches

Monday September 10th:  Full test set released. The public leaderboard will be locked and continue to reflect its state before the release of new data.  Daily model submission limits will be removed.  Visualization Prospect opens.

Monday, September 17th:  Up to five (5) final models must be submitted and locked-in (self-contained code).  Please see Rules for exact details on model submission.  Impermium will release the **verification set**

Thursday September 20th:  Deadline to submit up to five (5) solution on the verification set.  Solution must be generated with same code that was locked in on 9-17.

**The overall winners of the competition will be determined by their performance on the verification set.**  Winning entries will be checked to confirm that they were generated with the locked code.

Friday, September 21st:  Visualization prospect closes

### Winners

This challenge focused on using natural language processing to flag insulting comments posted in online forums.

- **First Place**: [Vivek Sharma](https://www.kaggle.com/users/5048/vivek-sharma), New Delhi, India
- **Second Place**: [Dmitry S](https://www.kaggle.com/users/45651/tuzzeg)., USA
- **Visualization Prize**: [Charlie Turner](https://www.kaggle.com/users/5940/charlie-turner), Davis, USA

### Citation

Cory O'Connor, Glider, parag. (2012). Detecting Insults in Social Commentary. Kaggle. https://kaggle.com/competitions/detecting-insults-in-social-commentary

# Data

## Dataset Description

### Data

The data consists of a label column followed by two attribute fields. 

This is a single-class classification problem. The label is either 0 meaning a **neutral** comment, or 1 meaning an **insulting** comment (neutral can be considered as not belonging to the insult class.  Your predictions must be a real number in the range [0,1] where 1 indicates 100% confident prediction that comment is an insult.

The first attribute is the time at which the comment was made. It is sometimes blank, meaning an accurate timestamp is not possible. It is in the form "YYYYMMDDhhmmss" and then the Z character. It is on a 24 hour clock and corresponds to the localtime at which the comment was originally made.

The second attribute is the unicode-escaped text of the content, surrounded by double-quotes. The content is mostly english language comments, with some occasional formatting.

### Guidelines

- We are looking for comments that are intended to be insulting to a person who is a part of the larger blog/forum conversation. 
- We are NOT looking for insults directed to non-participants (such as celebrities, public figures etc.). 
- Insults could contain profanity, racial slurs, or other offensive language. But often times, they do not. 
- Comments which contain profanity or racial slurs, but are not necessarily insulting to another person are considered not insulting.
- The insulting nature of the comment should be obvious, and not subtle. 

There may be a small amount of noise in the labels as they have not been meticulously cleaned. However, contenstants can be confident the error in the training and testing data is `< 1%`. 

Contestants should also be warned that this problem tends to strongly overfit. The provided data is generally representative of the full test set, but not exhaustive by any measure. Impermium will be conducting final evaluations based on an unpublished set of data drawn from a wide sample.
