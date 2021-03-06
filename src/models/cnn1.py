"""
A CNN model
"""

import torch
import torch.utils.data
from models.Model import Model


class CNN(Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.l1 = torch.nn.Sequential(
            # Conv Layer 1
            # image size: 3 x 128 x 128
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True),
            # image size: 3 x 124 x 124
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=True),  # inplace=True: to minimize memory usage
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
            torch.nn.BatchNorm1d(self.num_classes),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = self.l2(x)
        return x
