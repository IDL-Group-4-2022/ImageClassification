"""
This file trains a pytorch model, given the parameters and directory for
the data. It acts as a mediator (see mediator design pattern for details)
between the other classes, then saved the model to resources/models/
"""

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from utils.image_loader import get_dataloaders
from models.cnn2 import CNN as Model
# from models.transferred import Transferred as Model

torch.manual_seed(42)
np.random.seed(42)

# Hyper parameters
VERBOSE = 2
N_EPOCHS = 100
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 2000
LR = 0.05
WEIGHT_DECAY = 1e-5


# Constants
NUM_CLASSES = 14
IMG_DIR = 'resources/data/original/dl2021-image-corpus-proj/images'

df = pd.read_csv('resources/data/generated/train.csv', index_col='im_name')

# Dataset initialization
train_loader, dev_loader = get_dataloaders(
    df,
    test_size=0.2,
    img_dir=IMG_DIR,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_test=BATCH_SIZE_TRAIN
)

# GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# setup
model = Model(NUM_CLASSES).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # weight_decay adds l2 norm regularizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)  # weight_decay adds l2 norm regularizer

# ratio as per recommended: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
# This is needed because the classes are imbalanced
class_positive_counts = np.sum(df.values, axis=0)
class_negative_counts = len(df) - np.sum(df.values, axis=0)
ratio = class_negative_counts / class_positive_counts
weights = torch.Tensor(ratio).to(device)
weights = weights / sum(weights)

loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

# training
model.do_train(
    N_EPOCHS,
    device,
    loss_function,
    optimizer,
    train_loader,
    dev_loader,
    verbose=VERBOSE,
)

# torch.save(model, 'resources/models/Transferred.pytorch')
torch.save(model, 'resources/models/Cnn2.pytorch')
