## Description

The Multi-modal gesture recognition challenge, focused on gesture recognition from 2D and 3D video data using Kinect, is organized by ChaLearn in conjunction with ICMI 2013.

Kinect is revolutionizing the field of gesture recognition given the set of input data modalities it provides, including RGB image, depth image (using an infrared sensor), and audio. Gesture recognition is genuinely important in many multi-modal interaction and computer vision applications, including image/video indexing, video surveillance, computer interfaces, and gaming. It also provides excellent benchmarks for algorithms. The recognition of continuous, natural signing is very challenging due to the multimodal nature of the visual cues (e.g., movements of fingers and lips, facial expressions, body pose), as well as technical limitations such as spatial and temporal resolution and unreliable depth cues.

![banner_chalearn](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3386/media/banner_chalearn.png)

The Multi-modal Challenge workshop will be devoted to the presentation of most recent and challenging techniques from multi-modal gesture recognition. The committee encourages paper submissions in the following topics (but not limited to):

-   Multi-modal descriptors for gesture recognition
-   Fusion strategies for gesture recognition
-   Multi-modal learning for gesture recognition
-   Data sets and evaluation protocols for multi-modal gesture recognition
-   Applications of multi-modal gesture recognition

The results of the challenge will be discussed at the workshop. It features a quantitative evaluation of automatic gesture recognition from a multi-modal dataset recorded with Kinect (providing RGB images of face and body, depth images of face and body, skeleton information, joint orientation and audio sources), including about 15,000 Italian gestures from several users. The emphasis of this edition of the competition will be on multi-modal automatic learning of a vocabulary of 20 types of Italian gestures performed by several different users while explaining a history, with the aim of performing user independent continuous gesture recognition combined with audio information. 

Additionally, the challenge includes a live competition of demos/systems of applications based on multi-modal gesture recognition techniques. Demos using data from different modalities and different kind of devices are welcome. The demos will be evaluated in terms of multi-modality, technical quality, and applicability.

Best workshop papers and top three ranked participants of the quantitative evaluation will be invited to present their work at ICMI 2013 and their papers will be published in the ACM proceedings. Additionally, there will be travel grants (based on availability) and the possibility to be invited to present extended versions of their works to a special issue in a high impact factor journal. Moreover, all three top ranking participants in both, quantitative and qualitative challenges will be awarded with a ChaLearn winner certificate and an economic prize (based on availability). We will also announce a best paper and best student paper awards among the workshop contributions.

## Evaluation

The focus of the challenge is on "multiple instance, user independent learning" of gestures, which means learning to recognize gestures from several instances for each category performed by different users, drawn from a gesture vocabulary of 20 categories. A gesture vocabulary is a set of unique gestures, generally related to a particular task. In this challenge we will focus on the recognition of a vocabulary of 20 Italian cultural/anthropological signs.

### Challenge stages:

-   **Development phase**: Create a learning system capable of learning from several training examples a gesture classification problem. Practice with development data (a large database of 8,500 labeled gestures is available) and submit predictions on-line on validation data (3,500 labeled gestures) to get immediate feed-back on the leaderboard. Recommended: towards the end of the development phase, submit your code for verification purpose.
-   **Final evaluation phase**: Make predictions on the new final evaluation data (3,500 gestures) revealed at the end of the development phase. The participants will have few days to train their systems and upload their predictions.

We highly recommend that the participants take advantage of this opportunity and upload regularly updated versions of their code during the development period. Their last code submission before deadline will be used for the verification.

### What do you need to predict?

Each video contains the recording of multi-modal RGB-Depth-Audio data and user mask and skeleton information of several gesture instances from a vocabulary of 20 gesture categories of Italian signs.

You need to predict the identity of those gestures, represented by a numeric label (from 1 to 20).

In the data used for training, you get several video clips with annotated gesture labels for training purposes. Multiple gesture instances from several users will be available. In the data used for evaluation (validation data) you must predict the labels of the gestures played in a set of unlabeled videos.

Prediction is expected to be performed at gesture level, using the numeric label (1-20) for each frame with recognized gesture. The equivalence between gesture labels and gesture identifiers is provided in the data description page and a script for Matlab is also provided. The result will be a csv file with the video identifier and a coma separated list of recognized gestures (see example file provided with training data).

### Levenshtein Distance

For each video, you provide an ordered list of labels R corresponding to the recognized gestures. We compare this list to the corresponding list of labels T in the prescribed list of gestures that the user had to play. These are the "true" gesture labels (provided that the users did not make mistakes). We compute the so-called Levenshtein distance L(R, T), that is the minimum number of edit operations (substitution, insertion, or deletion) that one has to perform to go from R to T (or vice versa). The Levenhstein distance is also known as "edit distance".

For example:

    L([1 2 4], [3 2]) = 2

    L([1], [2]) = 1

    L([2 2 2], [2]) = 2

### Score

The overall score we compute is the sum of the Levenshtein distances for all the lines of the result file compared to the corresponding lines in the truth value file, divided by the total number of gestures in the truth value file. This score is analogous to an error rate. However, it can exceed one.

- **Public score** means the score that appears on the leaderboard during the development period and is based on the validation data.

- **Final score** means the score that will be computed on the final evaluation data released at the end of the development period, which will not be revealed until the challenge is over. The final score will be used to rank the participants and determine the prizes.

### Verification procedure

To verify that the participants complied with the rule that there should be no manual labelling of the test data, the top ranking participants eligible to win prizes will be asked to cooperate with the organizers to reproduce their results.

During the development period the participants can upload executable code reproducing their results together with their submissions. The organizers will evaluate requests to support particular platforms, but do not commit to support all platforms. The sooner a version of the code is uploaded, the highest the chances that the organizers will succeed in running it on their platform. The burden of proof will rest on the participants.

The code will be kept in confidence and used only for verification purpose after the challenge is over. The code submitted will need to be standalone and in particular it will not be allowed to access the Internet. It will need to be capable of training models from the final evaluation data training examples, for each data batch, and making label predictions on the test examples of that batch.

### Data split

We split the recorded data into:

- **training data**: fully labelled data that can be used for training and validation as desired.

- **validation data**: a dataset formatted in a similar way as the final evaluation data that can be used to practice making submissions on the Kaggle platform. The results on validation data will show immediately as the "public score" on the leaderboard.

- **final evaluation data**: the dataset that will be used to compute the final score (will be released shortly before the end of the challenge).

### Kaggle submission format

In order to submit your results to the Kaggle platform, you should provide a csv file with the following format:

Id,Sequence

0001,2 4 5 6 1

0002,1 2 12 4 14 16 3

where the first line is the header and the rest of the lines contains the predicted sequence of gestures. Notice that the Id correspond to the last 4 digits of the SampleID, that is:

Sample0**0001**.zip  =>  0001

A sample file corresponding to the training data is available for downloading.

## Prizes

### (a)    Incentive Prizes

(1)   ICMI 2013 competition:

*Quantitative competition:*

> -   First place: 500 USD and 1000 USD travel award + Award certificate
> -   Second place: 250 USD and 750 USD travel award + Award certificate
> -   Third place: 100 USD and 400 USD travel award + Award certificate

*Qualitative competition:*

> -   First place: 500 USD and 1000 USD travel award [based upon availability] + Award certificate
> -   Second place: 250 USD and 750 USD travel award [based upon availability] + Award certificate
> -   Third place: 100 USD and 400 USD travel award [based upon availability] + Award certificate

### (b)   Best paper awards:

CHALEARN will also give a best paper award and a best student paper award in both tracks, consisting of a CHALEARN certificate.

### (c)    Travel awards:

CHALEARN will also distribute travel awards to selected participants in the quantitative evaluation who also wish to participate in the qualitative live evaluation (to partially cover their travel expenses). The awards will be distributed according to merit, need, and availability.

### (d) If for any reason the advertised prize is unavailable, unless to do so would be prohibited by law, we reserve the right to substitute a prize(s) of equal or greater value, as permitted. We will only award one prize per team. If you are selected as a potential winner of this contest:

(1) If your prize is not in cash, you may not exchange your prize for cash; you may not exchange any prize for other merchandise or services.

(2) You may not designate someone else as the winner. If you are unable or unwilling to accept your prize, we will award it to an alternate potential winner.

(3) If you accept a prize, you will be solely responsible for all applicable taxes related to accepting the prize.

(4) If you are a minor in your place of residence, we may award the prize to your parent/legal guardian on your behalf and your parent/legal guardian will be designated as the winner.

(5) For South African residents, the value of the prize includes the value added tax (VAT).

## Timeline

### Quantitative Challenge

| Event | Date |
| --- | --- |
| Beginning of the quantitative competition, release of first data examples. | April 30th, 2013 |
| Full release of training and validation data. | May 20, 2013 |
| Release of final evaluation data. | August 1st, 2013 |
| End of the quantitative competition. Deadline for code submission. The organizers start the code verification by running it on the final evaluation data. | August 15th, 2013 |
| Release of final evaluation data decryption key. | August 20th, 2013 |
| Deadline for submitting the fact sheets and the prediction results on final evaluation data. | August 25th, 2013 |
| Release of the verification results to the participants for review. Top ranked participants are invited to follow the workshop submission guide for inclusion at ICMI proceedings. | September 1st, 2013 |

### Qualitative Challenge

| Event | Date |
| --- | --- |
| Deadline of a two-page summary of the live demo proposal | July 15th, 2013 |
| Acceptance notification of demos for qualitative competition | July 25th, 2013 |
| Submission of additional and supplementary information for demos | July 30th, 2013 |
| Round 1 evaluation of demos (just for feedback and recommendation to present or not the demo contribution as a workshop paper submission, final evaluation is performed at the first day of the workshop). | August 15th, 2013 |

### Workshop

| Event | Date |
| --- | --- |
| Workshop paper submission deadline (top three ranked quantitative and qualitative participants will be invited to submit their contribution as a paper submission to the workshop) | September 15th, 2013 |
| Notification of workshop paper acceptance | September 30th, 2013 |
| Camera ready of workshop papers | October 7th, 2013 |

### Citation

Ben Hamner, Isabelle, LoPoal, sescalera, xbaro. (2013). Multi-modal Gesture Recognition. Kaggle. https://kaggle.com/competitions/multi-modal-gesture-recognition

# Dataset Description

The data is also available [here](http://sunai.uoc.edu/chalearn/).

The focus of the challenge is on "multiple instance, user independent learning" of gestures, which means learning to recognize gestures from several instances for each category performed by different users, drawn from a gesture vocabulary of 20 categories. A gesture vocabulary is a set of unique gestures, generally related to a particular task. In this challenge we will focus on the recognition of a vocabulary of 20 Italian cultural/anthropological signs.

## Challenge stages

**Development Phase**: Create a learning system capable of learning from several training examples a gesture classification problem. Practice with training data (a large database of 7,754 manually labeled gestures is available) and submit predictions on-line on validation data (3,362 labelled gestures) to get immediate feed-back on the leaderboard.

**Final Evaluation Phase**: Make predictions on the new final evaluation data (around 3,000 gestures) revealed at the end of the development phase. The participants will have few days to train their systems and upload their predictions.

## Provided Files

Both for the development and final evaluation phase, the data will have the same format. We provide several ZIP files for each dataset, each file containing all the files for one sequence. The name of the ZIP file is assumed as the sequence identifier (eg. Sample00001, Sample00002, ...), and all their related files start with this SessionID:

-   SessionID_audio: Audio file.
-   SessionID_color: Video file with the RGB information.
-   SessionID_depth: Video file with the Depth information
-   SessionID_user: Video file with the user segmentation information
-   SessionID_data: Matlab file with a structure called Video, with the sequence information:
-   NumFrames: Number of frames of the sequence
-   FrameRate: Frame rate of the sequence
-   MaxDepth: Maximum depth value for depth data
-   Frames: Skeleton information information for each frame of the videos in Kinect format. More information is provided at the end of this section (Exporting the data).
-   Labels: Sequence gestures. It is only provided for the training data, on validation and test this field is empty. For each gesture, the initial frame, the last frame and its name are provided.
-   Please, note that we provide some Matlab scripts that work with the zipped file, therefore, we recommend to do not unzip the sequence files.

## What do you need to predict?

The development data contains the recording of multi-modal RGB-Depth-Audio data and user mask and skeleton information of 7,754 gesture instances from a vocabulary of 20 gesture categories of Italian signs.

For each sequence, it is expected to recognize each gesture of interest (that means, gestures from the list of 20 Italian gestures) and generate the list of seen gestures. For instance:

```
Session00001,2,12,3
```

where Session00001 is the sequence id (name of the ZIP file) and we predict that it contains first the gesture 2 (vieniqui), then the gesture 12 (cosatifarei) and finally the gesture 3 (perfetto). Be aware that sequence can contain non interest gestures (gestures that people perform but are not in the list of those 20 gestures), and those gestures can not be in the prediction list.

Next we list the types of gestures, represented by a numeric label (from 1 to 20), together with the number of training performances recorded for each gesture in brackets:

1.  'vattene' (389)
2.  'vieniqui' (390)
3.  'perfetto' (388)
4.  'furbo' (388)
5.  'cheduepalle' (389)
6.  'chevuoi' (387)
7.  'daccordo' (372)
8.  'seipazzo' (388)
9.  'combinato' (388)
10. 'freganiente' (387)
11. 'ok'  (391)
12. 'cosatifarei' (388)
13. 'basta' (388)
14. 'prendere' (390)
15. 'noncenepiu'  (383)
16. 'fame' (391)
17. 'tantotempo' (389)
18. 'buonissimo' (391)
19. 'messidaccordo' (389)
20. 'sonostufo' (388)

## Exporting the data

In the scripts section, we provide a Matlab GUI application that allows to view and listen the data and to export it to make easy to work. It exports the multimodal data in the following four Matlab structures:

-   NumFrames: Total number of frames.
-   FrameRate: Frame rate of a video.
-   Audio: structure that contains audio data.
-   y: audio data
-   fs: sample rate for the data
-   Labels: structure that contains the data about labels.
-   Name: The name given to this gesture.
-   Begin: It indicates the start frame of the gesture.
-   End: It indicates the ending frame of the gesture.
-   Label names are as follows:
    1.  'vattene'
    2.  'vieniqui'
    3.  'perfetto'
    4.  'furbo'
    5.  'cheduepalle'
    6.  'chevuoi'
    7.  'daccordo'
    8.  'seipazzo'
    9.  'combinato'
    10. 'freganiente'
    11. 'ok'
    12. 'cosatifarei'
    13. 'basta'
    14. 'prendere'
    15. 'noncenepiu'
    16. 'fame'
    17. 'tantotempo'
    18. 'buonissimo'
    19. 'messidaccordo'
    20. 'sonostufo'

After exportation, an individual mat (Sample00001_X.mat, where X indicates the number of the frame) file for each frame is generated containing the following structures:

-   RGB.
-   Depth.
-   UserIndex
-   Skeleton.

The detailed descriptions about each one of these structures are explained in the following section.

## On the generated MAT

A generated MAT file stores the selected visual data in three structures named 'RGB', 'Depth', 'UserIndex', and 'Skeleton'. The data of each structure is aligned regarding RGB data. The following subsections will describe each of these structures.

-   RGB data: this matrix represents the RGB color image.
-   Depth Frame: The Depth matrix contains the pixel-wise z-component of the point cloud. The value of depth is expressed in millimeters.
-   User Index: The User index matrix represents the player index of each pixel of the depth image. A non-zero pixel value signifies that a tracked subject occupies the pixel, and a value of 0 denotes that no tracked subject occupies the pixel.
-   Skeleton Frame: An array of Skeleton structures is contained within a Skeletons array. It contains the joint positions, and bone orientations comprising a skeleton. The format of a Skeleton structure is given below.
    -   Skeleton
    -   JointsType
    -   WorldPosition
    -   PixelPosition
    -   WorldRotation
-   JointsType can be as follows:
    -   HipCenter
    -   Spine
    -   ShoulderCenter
    -   Head
    -   ShoulderLeft
    -   ElbowLeft
    -   WristLeft
    -   HandLeft
    -   ShoulderRight
    -   ElbowRight
    -   WristRight
    -   HandRight
    -   HipLeft
    -   KneeLeft
    -   AnkleLeft
    -   FootLeft
    -   HipRight
    -   KneeRight
    -   AnkleRight
    -   FootRight
-   WorldPosition: The world coordinates position structure represents the global position of a tracked joint.
    -   The X value represents the x-component of the subject's global position (in millimeters).
    -   The Y value represents the y-component of the subject's global position (in millimeters).
    -   The Z value represents the z-component of the subject's global position (in millimeters).
-   PixelPosition: The pixel coordinates position structure represents the position of a tracked joint. The format of the Position structure is given as follows.
    -   The X value represents the x-component of the joint location (in pixels).
    -   The Y value represents the y-component of the joint location (in pixels).
    -   The Z value represents the z-component of the joint location (in pixels).
-   WorldRotation: The world rotation structure contains the orientations of skeletal bones in terms of absolute transformations. The world rotation structure provides the orientation of a bone in the 3D camera space. Is formed by 20x4 matrix, where each row contains the W, X, Y, Z values of the quaternion related to the rotation.
    -   The X value represents the x-component of the quaternion.
    -   The Y value represents the y-component of the quaternion.
    -   The Z value represents the z-component of the quaternion.
    -   The W value represents the w-component of the quaternion.