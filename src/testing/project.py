# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Team 4 - IDL - Project (image classification)
#
# Members:
#
# * Daniel Cauchi
# * Outi Savolainen
# * Silva Perander

# %%
import sys
sys.path += ["src"]

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.cnn1 import CNN as Model
from utils.image_loader import get_dataloaders

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
IMG_DIR = 'resources/data/original/dl2021-image-corpus-proj/images'
LABEL_DIR = 'resources/data/original/dl2021-image-corpus-proj/annotations'

# %%
df = pd.read_csv('resources/data/generated/train.csv', index_col='im_name')
# %%
# --- Dataset initialization ---

train_loader, dev_loader = get_dataloaders(
    df, 
    test_size=0.2, 
    img_dir=IMG_DIR, 
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_test=BATCH_SIZE_TRAIN
)

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%
#--- set up ---
model = Model(NUM_CLASSES).to(device)

# good until here :) yay

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # weight_decay adds l2 norm regularizer

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

        print('Dev: Epoch %d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                  (epoch + 1, loss_total,
                   100. * dev_correct / dev_total, dev_correct, dev_total))

        # # early stopping
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