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
from multiprocessing import reduction
import sys
sys.path += ["src"]

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from utils.image_loader import get_dataloaders
# from models.cnn1 import CNN as Model
from models.transferred import Transferred as Model
from models.cnn2 import CNN2 as Model2

torch.manual_seed(42)
np.random.seed(42)

# %%
#--- hyperparameters ---
VERBOSE = 2
N_EPOCHS = 100
BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST = 2000
LR = 0.01
WEIGHT_DECAY = 1e-5


# %%
#--- fixed constants ---
NUM_CLASSES = 14
IMG_DIR = 'resources/data/original/dl2021-image-corpus-proj/images'

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
model = Model2(NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # weight_decay adds l2 norm regularizer
#optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)  # weight_decay adds l2 norm regularizer

loss_function = torch.nn.BCELoss()

#--- training ---
model.do_train(
    N_EPOCHS,
    device,
    loss_function,
    optimizer,
    train_loader,
    dev_loader,
    verbose=VERBOSE,
)

# %%
#--- test ---
# total = 0
# total_correct = 0

# with torch.no_grad():
#     for batch_num, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         loss = loss_function(output, target)
#         test_loss = loss.item()
#         predicted = torch.argmax(output, axis=1)
#         correct = sum(predicted == target)
#         total_correct += correct
#         count = len(target)
#         total += count
#         print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
#               (batch_num + 1, len(test_loader), test_loss / (batch_num + 1),
#                100. * correct / count, correct, count))

#     print(f'Total Test Accuracy: {total_correct}/{total} = {total_correct / total * 100}%')
