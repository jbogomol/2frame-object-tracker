2frame-object-tracker

A frame-to-frame object tracker designed using pytorch and trained on images
created from the flickr30k dataset. Given two frames, predict the x and y
offset of an object from one frame to the next.

By Jackson Bogomolny <jbogomol@andrew.cmu.edu>
Research project for Prof. Aswin Sankaranarayanan of Carnegie Mellon University
Electrical & Computer Engineering department



~ IMPLEMENTATION DETAILS ~

How the dataset is created:
  - Resize all flickr30k images to 256x256.
  - Randomly select 10,000 images to be "backgrounds" and another 10,000 to be
    "foregrounds."
  - For each background image, place an "object" (128x128 cutout) from a
    foreground image in the center of the background image. This is frame 1.
  - Create frame 2 the same way as frame 1, but with the object shifted from
    the center by a random vector v = <vx, vy> such that:
        -10 <= vx <= 10
        -10 <= vy <= 10
    where vx is the horizontal shift from the center, and vy is the vertical
    shift from the center.
  - A batch of size 1 includes both corresponding frames and their vector v.

Network architecture:
  - Input layer of shape torch.Size([<batch_size>, 6, 256, 256])
        - The 6 channels are the RGB channels of both frames concatenated
  - 3 convolutional layers with the following specs:
        1. 3x3 kernel, 16 feature maps, 2x2 stride
           conv, batch norm, relu
        2. 3x3 kernel, 32 feature maps, 2x2 stride
           conv, batch norm, relu
        3. 5x5 kernel, 64 feature maps, 2x2 stride
           conv, batch norm, relu
  - 3 fully-connected layers with the following specs:
        1. size 120, relu
        2. size 60,  relu
        3. size 42,  relu (output layer) - 2frame_tracking_classifier.py
           output is read by reshaping this layer to
           torch.Size([<batch_size>, 2, 21]), where the 2 axes of length 21
           represent the probabilities of vx=<index> and vy=<index>
           (i.e. the argmax of the output layer is the one-hot encoded
           prediction of the object's offset vector v).
           -OR-
           size 2 (output layer) - 2frame_tracking_regression.py
           where the first number is the predicted vx,
           and the second number is the predicted vy



~ HOW TO RUN ~

1. resize_all.py
    Takes images from source directory, resizes them to a given
    size, and saves the results in a specified results directory.
2. create_images.py
    Takes resized images and creates a results directory containing
    10,000 2-frame pairs (20,000 jpg images) and a results.csv file
    containing paths to the images and their corresponding offset vector's
    components, vx and vy.
3. 2frame_tracking_classifier.py
    Creates pytorch dataset and dataloader, trains the network, and reports
    the network's progress on validation data to the command line.
    Formulated as a classification problem, trained with a cross-entropy loss.
   -OR-
   2frame_tracking_regression.py
    Formulated as a regression problem, trained with a regression loss
    function.

For all
    Change global vars to correct filepaths, image sizes, etc.
    Usage:
    >  python <filename.py>



~ DEPENDENCIES ~

python:
    python 3.7.3

all packages:
    <package> (<my version>)
    torch (1.5.0)
    torchvision (0.6.0a0+82fd1c8)
    pandas (0.24.2)
    cv2 (4.2.0)
    numpy (1.16.2)
    matplotlib (3.1.3)
    seaborn (0.10.0)
    os
    sys
    random

dataset:
    Flickr30k
    http://shannon.cs.illinois.edu/DenotationGraph/data/index.html
    P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. From image
    description to visual denotations: New similarity metrics for semantic
    inference over event descriptions. Transactions of the Association for
    Computational Linguistics (to appear).






