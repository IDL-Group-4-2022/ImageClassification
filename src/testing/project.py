# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Team 4 - IDL - Exercise 2
# 
# Members:
# 
# * Daniel Cauchi
# * Outi Savolainen
# * Silva Perander
# %% [markdown]
# # Starting Notes
# 
# 
# * I changed the batch sizes for train and test to increase speed
# * Results of our experiments are at the end of the notebook as markdown cells

# %%
import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from os import listdir
import os
from torchvision.io import read_image
from image_loader import CustomImageDataSet
from sklearn.model_selection import train_test_split


torch.manual_seed(42)
np.random.seed(42)


# %%
#--- hyperparameters ---
VERBOSE = True
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 2000
LR = 0.05
WEIGHT_DECAY = 1e-5


# %%
#--- fixed constants ---
NUM_CLASSES = 14
IMG_DIR = 'dl2021-image-corpus-proj/images'
LABEL_DIR = 'dl2021-image-corpus-proj/annotations'

img = cv2.imread(f"{IMG_DIR}/im1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
plt.imshow(img)


# %%

label_files = [f for f in listdir(LABEL_DIR)]
label_names = [os.path.splitext(f)[0] for f in label_files]

img_labels = {}

for f in label_files:
    label = os.path.splitext(f)[0]
    with open(os.path.join(LABEL_DIR, f)) as fi:

        ids = fi.readlines()

        for i in ids:
            # print(i)
            i = int(i)
            if i not in img_labels:
                img_labels[i] = []
            img_labels[i].append(label)

# print(img_labels)

# %%
df = pd.DataFrame([img_labels], index=['labels']).T
df = df.drop('labels', 1).join(df.labels.str.join('|').str.get_dummies())
print(df.head(5))
print(np.array(df.columns))
print(np.array(df.index))
print(len(df))

#%%
idx = 0

img_name = df.index[idx]
print(img_name)
img_path = os.path.join(IMG_DIR, f"im{img_name}.jpg")
print(img_path)
image = read_image(img_path)
label = np.array(df.iloc[[idx]])[0]
print(label)

# %%
# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
colour_jitter = 0.2
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=colour_jitter,
        contrast=colour_jitter,
        saturation=colour_jitter,
        hue=colour_jitter,
    ),
    transforms.RandomAffine(degrees=10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 4)),
    transforms.ToTensor(),
])
grayscale_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    ])
test_transform = transforms.Compose([transforms.ToTensor()])

train, dev = train_test_split(df, test_size=0.2)

train_set = CustomImageDataSet(train, IMG_DIR, transform=train_transform, grayscale_transform=grayscale_transform)
dev_set = CustomImageDataSet(dev, IMG_DIR, transform=test_transform, grayscale_transform=grayscale_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

# %% [markdown]
# docs to calculate image sizes after convolution and max pooling 2d:
# * https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# * https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

# %%
#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.l1 = torch.nn.Sequential(
            # Conv Layer 1
            # image size: 3 x 128 x 128
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True),
            # image size: 3 x 124 x 124
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=True), # inplace=True: to minimize memory usage
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # image size: 3 x 62 x 62
            
            # Conv layer 2
            torch.nn.Conv2d(in_channels=6, out_channels=7, kernel_size=3, stride=1, padding=0, bias=True),
            # image size: 3 x 60 x 60
            torch.nn.BatchNorm2d(7),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # image size: 3 x 30 x 30

            # Conv layer 3
            torch.nn.Conv2d(in_channels=7, out_channels=6, kernel_size=3, stride=1, padding=0, bias=True),
            # image size: 3 x 28 x 28
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # image size: 3 x 14 x 14

            # Conv layer 4
            torch.nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3, stride=1, padding=0, bias=True),
            # image size: 3 x 12 x 12
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # image size: 3 x 6 x 6
        )

        self.l2 = torch.nn.Sequential(
            torch.nn.Linear(4 * 6 * 6, 4 * 7 * 7),
            torch.nn.BatchNorm1d(4 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * 7 * 7, 4 * 7 * 7),
            torch.nn.BatchNorm1d(4 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * 7 * 7, self.num_classes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = self.l2(x)
        return x


# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %% [markdown]
# ## Training with Adam Optimizer

# %%
#--- set up ---
model = CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # weight_decay adds l2 norm regularizer

# loss_function = torch.nn.NLLLoss()
# loss_function = torch.nn.MultiLabelSoftMarginLoss(reduction='none')
loss_function = torch.nn.BCELoss()

losses = []
#--- training ---
previous_loss_total = 99999999999
for epoch in range(N_EPOCHS):
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)

        output = output.to(torch.float32)
        target = target.to(torch.float32)

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        total = target.shape[0] * target.shape[1]
        train_loss = loss.item()

        predicted = torch.where(output > 0.5, 1, 0)

        train_correct = torch.sum(predicted == target)

        losses.append(train_loss)

        if VERBOSE:
            
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch + 1, batch_num + 1, len(train_loader), train_loss, 
                   100. * train_correct / total, train_correct, total))
    
        plt.plot(losses)
        plt.savefig("losses_adam_multilabel.png")


    with torch.no_grad():
        dev_total = 0
        dev_correct = 0
        loss_total = 0
        for dev_data, dev_target in dev_loader:
            dev_data, dev_target = dev_data.to(device), dev_target.to(device)
            
            dev_output = model(dev_data)
            
            dev_output = dev_output.to(torch.float32)
            dev_target = dev_target.to(torch.float32)
            
            predicted = torch.where(dev_output > 0.5, 1, 0)
            
            dev_correct += torch.sum(predicted == dev_target)
            dev_total += dev_target.shape[0] * dev_target.shape[1]
            loss_total += loss_function(dev_output, dev_target).item()
        # print(

        #     f'Dev accuracy for epoch {epoch  + 1}:'
        #     f'{dev_correct}/{dev_total} = {dev_correct / dev_total * 100}%\n'
        #     f'Loss: {loss_total}\n\n'
        # )
        print('Dev: Epoch %d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch + 1, loss_total, 
                   100. * dev_correct / dev_total, dev_correct, dev_total))
    
        # if previous_loss_total < loss_total:
        #     break
        # previous_loss_total = loss_total


# %%
#--- test ---
total = 0
total_correct = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        test_loss = loss.item()
        predicted = torch.argmax(output, axis=1)
        correct = sum(predicted == target)
        total_correct += correct
        count = len(target)
        total += count
        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num + 1, len(test_loader), test_loss / (batch_num + 1), 
               100. * correct / count, correct, count))
        
    print(f'Total Test Accuracy: {total_correct}/{total} = {total_correct / total * 100}%')

# %% [markdown]
# ## Training with Adagrad Optimizer

# %%
#--- set up ---
model = CNN().to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=LR,  weight_decay=WEIGHT_DECAY)  # weight_decay adds l2 norm regularizer

loss_function = torch.nn.NLLLoss()

losses = []

#--- training ---
previous_loss_total = 99999999999
for epoch in range(N_EPOCHS):
    train_loss = 0
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        # Optimize
        loss.backward()
        optimizer.step()

        total = len(target)
        train_loss += loss.item()

        predicted = torch.argmax(output, axis=1)
        train_correct = sum(predicted == target)
        
        if VERBOSE:
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch + 1, batch_num + 1, len(train_loader), train_loss / (batch_num + 1), 
                   100. * train_correct / total, train_correct, total))
    
        losses.append(train_loss/(batch_num + 1))
        plt.plot(losses)
        plt.savefig("losses_adagrad_avgperbatch.png")

    # with torch.no_grad():
    #     # early stopping
    #     dev_total = 0
    #     dev_correct = 0
    #     loss_total = 0
    #     for dev_data, dev_target in dev_loader:
    #         dev_data, dev_target = dev_data.to(device), dev_target.to(device)
    #         dev_output = model(dev_data)
    #         predicted = torch.argmax(dev_output, axis=1)
    #         dev_correct += sum(predicted == dev_target)
    #         dev_total += len(dev_target)
    #         loss_total += loss_function(dev_output, dev_target)
    #     print(
    #         f'Dev accuracy for epoch {epoch  + 1}:'
    #         f'{dev_correct}/{dev_total} = {dev_correct / dev_total * 100}%\n'
    #         f'Loss: {loss_total}\n\n'
    #     )
    #     if previous_loss_total < loss_total:
    #         break
    #     previous_loss_total = loss_total


# %%
#--- test ---
total = 0
total_correct = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        test_loss = loss.item()
        predicted = torch.argmax(output, axis=1)
        correct = sum(predicted == target)
        total_correct += correct
        count = len(target)
        total += count
        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num + 1, len(test_loader), test_loss / (batch_num + 1), 
               100. * correct / count, correct, count))
        
    print(f'Total Test Accuracy: {total_correct}/{total} = {total_correct / total * 100}%')

# %% [markdown]
# ## Training with RMSProp Optimizer

# %%
#--- set up ---
model = CNN().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)  # weight_decay adds l2 norm regularizer

loss_function = torch.nn.NLLLoss()
#--- training ---
previous_loss_total = 99999999999
for epoch in range(N_EPOCHS):
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        # Optimize
        loss.backward()
        optimizer.step()

        total = len(target)
        train_loss = loss.item()
        predicted = torch.argmax(output, axis=1)
        train_correct = sum(predicted == target)
        
        if VERBOSE:
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch + 1, batch_num + 1, len(train_loader), train_loss / (batch_num + 1), 
                   100. * train_correct / total, train_correct, total))
    
    with torch.no_grad():
        # early stopping
        dev_total = 0
        dev_correct = 0
        loss_total = 0
        for dev_data, dev_target in dev_loader:
            dev_data, dev_target = dev_data.to(device), dev_target.to(device)
            dev_output = model(dev_data)
            predicted = torch.argmax(dev_output, axis=1)
            dev_correct += sum(predicted == dev_target)
            dev_total += len(dev_target)
            loss_total += loss_function(dev_output, dev_target)
        print(
            f'Dev accuracy for epoch {epoch  + 1}:'
            f'{dev_correct}/{dev_total} = {dev_correct / dev_total * 100}%\n'
            f'Loss: {loss_total}\n\n'
        )
        if previous_loss_total < loss_total:
            break
        previous_loss_total = loss_total


# %%
#--- test ---
total = 0
total_correct = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        test_loss = loss.item()
        predicted = torch.argmax(output, axis=1)
        correct = sum(predicted == target)
        total_correct += correct
        count = len(target)
        total += count
        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num + 1, len(test_loader), test_loss / (batch_num + 1), 
               100. * correct / count, correct, count))
        
    print(f'Total Test Accuracy: {total_correct}/{total} = {total_correct / total * 100}%')

# %% [markdown]
# # Regularizers and Optimizers
# 
# With regards to regularization, we are using:
# * Data augmentation (in the train_transform when loading the data)
# * Early stopping
# * Dropout (in the linear layers of the model)
# * L2 norm (the weight_decay parameter of the optimizers)
# 
# With regards to optimization, we have experimented with these:
# * Batch Normalization (for both convolutional and linear layers)
# * Momentum for SGD
# * Different optimizers:
#     * Adagrad
#     * Adam
#     * SGD
# %% [markdown]
# # Concluding Thoughts
# 
# * Using too much regularization will lead to very slow convergence. For example:
#     * Using too much dropout
#     * Setting weight decay to a too high value
#     
# * Faster convergence was achieved with a high batch size of 500, when compared to one with 100
#     * In some cases, with a small batch size (100) and more output channels of the convolutional layers, it could lead to faster convergence.
# 
# * Batch normalization helped tremendously in convergence. Some models often fell into local minima without it
# 
# * Best accuracy was obtained with the **Adam optimizer**.
#     * The accuracy score is: 74.6%

