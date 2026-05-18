# Task

Identify the gestures made in a video. For each video, provide an ordered list of labels R corresponding to the recognized gestures. 

Each video contains the recording of multi-modal RGB-Depth-Audio data and user mask and skeleton information of several gesture instances from a vocabulary of 20 gesture categories of Italian signs.

You need to predict the identity of those gestures, represented by a numeric label (from 1 to 20).

# Metric

Levenshtein distances for all the lines of the result file compared to the corresponding lines in the truth value file, divided by the total number of gestures in the truth value file. This score is analogous to an error rate. However, it can exceed one.

The focus of the challenge is on "multiple instance, user independent learning" of gestures, which means learning to recognize gestures from several instances for each category performed by different users, drawn from a gesture vocabulary of 20 categories. A gesture vocabulary is a set of unique gestures, generally related to a particular task. In this challenge we will focus on the recognition of a vocabulary of 20 Italian cultural/anthropological signs.

# Submission Format

For each sequence, it is expected to recognize each gesture of interest (that means, gestures from the list of 20 Italian gestures) and generate the list of seen gestures. For instance:

```
Session00001,2,12,3
```

where Session00001 is the sequence id (name of the ZIP file) and we predict that it contains first the gesture 2 (vieniqui), then the gesture 12 (cosatifarei) and finally the gesture 3 (perfetto). Be aware that sequence can contain non interest gestures (gestures that people perform but are not in the list of those 20 gestures), and those gestures can not be in the prediction list.

# Dataset

In the training data, you get several video clips with annotated gesture labels for training purposes.

We provide several ZIP files for each dataset, each file containing all the files for one sequence. The name of the ZIP file is assumed as the sequence identifier (eg. Sample00001, Sample00002, ...), and all their related files start with this SessionID:

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