# create-images.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         06/09/2020
#
# make 20k images from source imdir
# - 10k have fg placed in center of bg
# - 10k have fg displaced by random vector <vx, vy> where
#       vx and vy random ints in range [-maxtrans, maxtrans]
# 
# store resulting images in resultsdir
#       as 0c.jpg 0t.jpg ... 9999c.jpg 9999t.jpg
# store ground truths in resultsdir/results.csv file
#       with following columns: img_center,img_trans,vx,vy
#
# before running this, it is recommended to run:
#      resize-all.py
#      which will resize all images to the same size


import os
import cv2
import random
import pandas as pd


# directory with raw image data (must have >= 20k jpg images)
# all images should be same size for nn
# imdir = './flickr30k/resized'
imdir = '/home/datasets/data_jbogomol/flickr30k/resized'

# directory to store results in
# resultsdir = './flickr30k/results'
resultsdir = '/home/datasets/data_jbogomol/flickr30k/results'

# size of square "object" to cut from fg and paste in bg
objsize = 128

# max value the object can be translated by (10 means -10 to 10 inclusive)
maxtrans = 10

# display images as theyre made (press space to advance to next image)
display = False


# empty resultsdir
for filename in os.listdir(resultsdir):
    os.remove(os.path.join(resultsdir, filename))


# make lists of file paths for 10k bg images and 10k fg images
jpgs_seen = 0
bglist = []
fglist = []
for filename in os.listdir(imdir):
    if jpgs_seen == 20000:
        break
    elif filename.endswith('.jpg'):
        jpgs_seen += 1
        path = os.path.join(imdir, filename)
        if jpgs_seen <= 10000:
            bglist.append(path)
        else:
            fglist.append(path)

# make sure there were 20,000 jpg images, split evenly between bg and fg lists
if jpgs_seen != 20000 or len(bglist) != 10000 or len(fglist) != 10000:
    print('jpgs_seen (should be 20,000):', jpgs_seen)
    print('bglist size (should be 10,000):', len(bglist))
    print('fglist size (should be 10,000):', len(fglist))

# initialize results.csv file
csv = open(os.path.join(resultsdir, 'results.csv'), 'w+')
csv.write('img_center,img_trans,vx,vy\n')

# make the new images
halfobjsize = objsize // 2
for i in range(10000):
    # load a bg and fg image
    imgbg = cv2.imread(bglist[i], -1)
    imgfg = cv2.imread(fglist[i], -1)

    # make sure the images exist and have 3 (BGR) channels
    h_bg, w_bg, channels_bg = imgbg.shape
    h_fg, w_fg, channels_fg = imgfg.shape
    if channels_bg != 3:
        print(filename_bg + ' has ' + str(channels_bg) + ' channels!')
        continue
    if channels_fg != 3:
        print(filename_fg + ' has ' + str(channels_fg) + ' channels!')
        continue
    
    # get random translation vector
    vy = random.randint(-maxtrans, maxtrans)
    vx = random.randint(-maxtrans, maxtrans)

    # find center of both images
    cy_bg = h_bg // 2
    cx_bg = w_bg // 2
    cy_fg = h_fg // 2
    cx_fg = w_fg // 2

    # get coordinates of object
    # bg
    y1_bg = cy_bg - halfobjsize
    y2_bg = cy_bg + halfobjsize
    x1_bg = cx_bg - halfobjsize
    x2_bg = cx_bg + halfobjsize
    # fg
    y1_fg = cy_fg - halfobjsize
    y2_fg = cy_fg + halfobjsize
    x1_fg = cx_fg - halfobjsize
    x2_fg = cx_fg + halfobjsize
    # translated by vector
    y1_bgt = y1_bg + vy
    y2_bgt = y2_bg + vy
    x1_bgt = x1_bg + vx
    x2_bgt = x2_bg + vx

    # snip object from fg
    obj = imgfg[y1_fg:y2_fg, x1_fg:x2_fg]

    # paste object in center of bg
    imgobjcenter = imgbg.copy()
    imgobjtrans = imgbg.copy()
    imgobjcenter[y1_bg:y2_bg, x1_bg:x2_bg] = obj
    imgobjtrans[y1_bgt:y2_bgt, x1_bgt:x2_bgt] = obj

    # save new images to results directory
    path_center = os.path.join(resultsdir, str(i) + 'c.jpg')
    path_trans = os.path.join(resultsdir, str(i) + 't.jpg')
    cv2.imwrite(path_center, imgobjcenter)
    cv2.imwrite(path_trans, imgobjtrans)

    # append line in csv file
    csv.write(path_center + ','
              + path_trans + ','
              + str(vx) + ','
              + str(vy) +'\n')

    # display images if display global is set
    if display:
        cv2.imshow('fg image', imgfg)
        cv2.imshow('fg object', obj)
        cv2.imshow('bg image: before', imgbg)
        cv2.imshow('bg image: object in center', imgobjcenter)
        cv2.imshow('bg image: object translated by v = <'
                   + str(vx) + ', ' + str(vy) + '>', imgobjtrans)
        while cv2.waitKey(0) != 32: # spacebar
            pass
        cv2.destroyAllWindows()
    




