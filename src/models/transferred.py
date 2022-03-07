"""
A model with pretrained weights from another model
"""


import torch
import torch.utils.data
from torchvision import models
from models.Model import Model


class Transferred(Model):
    def __init__(self, num_classes):
        super(Transferred, self).__init__()
        # Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        self.num_classes = num_classes
        self.model = models.googlenet(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        hidden_layer_size = 20
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, hidden_layer_size),
            torch.nn.BatchNorm1d(hidden_layer_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.BatchNorm1d(hidden_layer_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.BatchNorm1d(hidden_layer_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_layer_size, num_classes),
            torch.nn.BatchNorm1d(num_classes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
