# 2frame_tracking_classifier.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/07/2020
#
# Trains a convolutional network to guess frame-to-frame object motion
# from -32 to 32 pixels in x or y direction. Interpolates with non-integer
# offset.
# Formulated as a regression problem with a mean squared error loss function.
#
# Before running, run create_images.py to create necessary images from
# data as well as results.csv file containing ground truths


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import cv2
import sys
import random


# train or not
training_on = True

# on server or local computer
on_server = torch.cuda.is_available()

# directory with images and results.csv
if on_server:
    resultsdir = '/home/datasets/data_jbogomol/flickr30k/results'
else:
    resultsdir = '../flickr30k/results'

# directory to store saves
reportdir = './report_regression/'

# csv path
csvpath = os.path.join(resultsdir, 'results.csv')

# network to save (if training) or load (if not training)
netpath = os.path.join(reportdir, '2frame_net.pth')

# which gpu
if on_server:
    torch.cuda.set_device(2)


# dataset class
class TwoFrameTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, csvpath, resultsdir, transform=None):
        self.resultsframe = pd.read_csv(csvpath)
        self.resultsdir = resultsdir
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        centerpath = self.resultsframe.iloc[index, 0]
        transpath = self.resultsframe.iloc[index, 1]
        imgcenter = torch.from_numpy(cv2.imread(centerpath, -1))
        imgtrans = torch.from_numpy(cv2.imread(transpath, -1))
        
        # change dimensions to match MNIST shape:
        #       ([batch size, channels, width, height])
        imgcenter = imgcenter.permute(2, 0, 1)
        imgtrans = imgtrans.permute(2, 0, 1)

        imgs = torch.cat((imgcenter, imgtrans), dim=0)
        imgs = imgs.float()
        
        v = self.resultsframe.iloc[index, 2:].to_numpy(dtype=float)
        
        data = [imgs, v]

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.resultsframe)


# network hyperparameters
n_epochs = 50
batch_size_train = 64
batch_size_validation = 1
batch_size_test = 1
learning_rate = 0.001 # 0.001 for classifier
momentum = 0.9
log_interval = 10 # print every log_interval mini batches

# keep same random seed for replicable results
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# train, validation, and test sets
fullset = TwoFrameTrackingDataset(
    csvpath=csvpath,
    resultsdir=resultsdir,
    transform=None)

n_train = 8000
n_validation = 1000
n_test = 1000
if n_train + n_validation + n_test > 10000:
    sys.exit('ERROR: size of train + validation + test sets must be <= 10,000'
             + ' (total size of the dataset).\nExiting...')

trainset = torch.utils.data.Subset(
    fullset, range(n_train))
validationset = torch.utils.data.Subset(
    fullset, range(n_train, n_train + n_validation))
testset = torch.utils.data.Subset(
    fullset, range(n_train + n_validation, n_train + n_validation + n_test))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_train, shuffle=True)
validationloader = torch.utils.data.DataLoader(
    validationset, batch_size=batch_size_validation, shuffle=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size_test, shuffle=True)

# check the shapes
dataiter = iter(trainloader)
imgs, v = dataiter.next()
print('imgs.shape:', imgs.shape)
print('v.shape:', v.shape)


# define network class (extends Module class)
class Network(nn.Module):
    def __init__(self):
        # constructor
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Linear(in_features=64*31*31, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=2)

    def forward(self, t):
        # (1) convolutional layer
        t = self.conv1(t)
        t = self.conv1_bn(t)
        t = F.relu(t)

        # (2) convolutional layer
        t = self.conv2(t)
        t = self.conv2_bn(t)
        t = F.relu(t)

        # (3) convolutional layer
        t = self.conv3(t)
        t = self.conv3_bn(t)
        t = F.relu(t)

        #print('after conv layers: t.shape =', t.shape)
        
        # (3) fully connected layer
        t = t.reshape(-1, 64*31*31)
        t = self.fc1(t)
        t = F.relu(t)

        # (4) fully connected layer
        t = self.fc2(t)
        t = F.relu(t)

        # (5) output layer
        t = self.out(t)
        return t

network = Network()
if on_server:
    network = network.cuda()
print('network:', network)

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

if training_on:
    # training loop
    print('Begin training loop')

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            # get inputs and labels
            inputs, labels = batch
            if on_server:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # reformat labels
            labels_x = labels[:,0].float()
            labels_y = labels[:,1].float()

            # zero the gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = network(inputs)
            outputs_x = outputs[:,0]
            outputs_y = outputs[:,1]
            loss_x = F.mse_loss(outputs_x, labels_x, reduction='sum')
            loss_y = F.mse_loss(outputs_y, labels_y, reduction='sum')
            loss = loss_x + loss_y
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if i % log_interval == (log_interval - 1):
                print('epoch %d,\tbatch %5d,\ttraining loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / log_interval))
                running_loss = 0.0

        print('end of epoch '+ str(epoch + 1))
    
        # test on validation set
        print('testing on validation set:')
        correct = 0
        off_by_one = 0
        total = n_validation * 2
        with torch.no_grad():
            for batch in validationloader:
                # should only be one batch
                images, labels = batch
                if on_server:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = network(images)
                preds = outputs
                labels = labels
                x_diff = preds[:,0] - labels[:,0]
                y_diff = preds[:,1] - labels[:,1]
                for i in range(batch_size_validation):
                    error_x = abs(x_diff[i].item())
                    error_y = abs(y_diff[i].item())

                    if error_x < 1:
                        correct += 1
                    elif error_x < 2:
                        off_by_one += 1
                    
                    if error_y < 1:
                        correct += 1
                    elif error_y < 2:
                        off_by_one += 1


        print('# correct:  ' + str(correct) + '/' + str(total) + ' = '
              + str(100.0*correct/total) + '%')
        print('# off by 1: ' + str(off_by_one) + '/' + str(total) + ' = '
              + str(100.0*off_by_one/total) + '%')

    print('Finished training')
    
    # save network
    torch.save(network.state_dict(), netpath)


# load network
if on_server:
    network.load_state_dict(
        torch.load(netpath, map_location=torch.device('cuda')))
else:
    network.load_state_dict(
        torch.load(netpath, map_location=torch.device('cpu')))


# test the trained network on test set (new data)
print('Testing network on test set')
correct = 0
errcount = 0
total = n_test * 2
heatmap = []
errmap = np.zeros([65, 65])

# empty error directory to fill with new errors
for filename in os.listdir(reportdir + 'errors/'):
    os.remove(os.path.join(reportdir + 'errors/', filename))

with torch.no_grad():
    for batch in testloader:
        images, labels = batch
        if on_server:
            images = images.cuda()
            labels = labels.cuda()
        outputs = network(images)
        preds = outputs
        predsint = preds.int()
        x_diff = preds[:,0] - labels[:,0]
        y_diff = preds[:,1] - labels[:,1]
        for i in range(batch_size_test):
            heatmap.append([x_diff[i].item(), y_diff[i].item()])
            error_x = abs(x_diff[i].item())
            error_y = abs(y_diff[i].item())
            error = error_x + error_y
            errmap[predsint[i][1]][predsint[i][0]] += error

            if error_x < 1:
                correct += 1
            if error_y < 1:
                correct += 1

            if error > 1:
                # if vector v predicted incorrectly
                # get actual and predicted vx,vy
                pred = preds[i]
                label = labels[i]
                vxp = pred[0].int().item()
                vyp = pred[1].int().item()
                vx = label[0].int().item()
                vy = label[1].int().item()
                # save an image with bounding boxes
                # actual in green, predicted in blue
                images = images.cpu()
                img = images[0].numpy()
                img_trans = img[3:,:,:]
                img_trans = np.transpose(img_trans, (1, 2, 0))
                img_trans = cv2.resize(img_trans, (256, 256))
                img_rect = cv2.rectangle(
                    img=img_trans,
                    pt1=(64+vx, 64+vy),
                    pt2=(64+128+vx, 64+128+vy),
                    color=(0, 255, 0),
                    thickness=1)
                img_rect = cv2.rectangle(
                    img=img_rect,
                    pt1=(64+vxp, 64+vyp),
                    pt2=(64+128+vxp, 64+128+vyp),
                    color=(255, 0, 0),
                    thickness=1)
                imgname = 'err_' + str(errcount) + '.jpg'
                cv2.imwrite(os.path.join(reportdir + 'errors/', imgname),
                    img_rect)
                errcount += 1

print('# correct:  ' + str(correct) + '/' + str(total) + ' = '
      + str(100.0*correct/total) + '%')
print(str(errcount) + ' errors saved to ' + reportdir + 'errors/')
print('predicted box in blue, correct in green')

# show heat map on testing data
heatmap = np.array(heatmap)
columns = ['X diff (prediction - label)', 'Y diff (prediction - label)']
heatmap_df = pd.DataFrame(data=heatmap, columns=columns)
plt.figure()
plt.title('Heat map, trained with mean squared error loss')
heatmap_plot = sb.jointplot(
        x='X diff (prediction - label)',
        y='Y diff (prediction - label)',
        data=heatmap_df,
        kind='scatter'
    )
plt.savefig(os.path.join(reportdir, 'heatmap_regression.png'))

# show error map
plt.figure()
plt.imshow(errmap, cmap='hot', interpolation='nearest')
plt.xlabel('x-component of offset vector')
plt.ylabel('y-component of offset vector')
plt.title('Sum of |ex| + |ey| on all images in test set')
plt.xticks(range(65), range(-32, 33))
plt.yticks(range(65), range(-32, 33))
plt.savefig(os.path.join(reportdir, 'errmap_regression.png'))


















