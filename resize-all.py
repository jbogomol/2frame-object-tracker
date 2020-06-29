# resize-all.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         06/11/2020
#
# resizes all images in a specified directory to a specified size


import os
import cv2


# directory with images
# imdir = './flickr30k/flickr30k_images'
imdir = '/home/datasets/data_jbogomol/flickr30k/flickr30k_images'

# directory for results
# resultsdir = './flickr30k/resized'
resultsdir = '/home/datasets/data_jbogomol/flickr30k/resized'

# size of results
resultssize = 256


# empty resultsdir
for filename in os.listdir(resultsdir):
    os.remove(os.path.join(resultsdir, filename))


# loop through imdir and crop all files to 256
for filename in os.listdir(imdir):
    if filename.endswith('.jpg'):
        filepath = os.path.join(imdir, filename)
        img = cv2.imread(filepath, -1)
        imgnew = cv2.resize(img, (resultssize, resultssize))
        filenamenew = str(resultssize) + '_' + filename
        cv2.imwrite(os.path.join(resultsdir, filenamenew), imgnew)
