"""
Parent class for our models which contains the training
"""

import torch.nn as nn
import torch
from matplotlib import pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def train(
        self,
        n_epochs: int,
        device: str,
        loss_function: torch.nn.Module,
        optimizer: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        dev_loader: torch.utils.data.DataLoader,
        VERBOSE: bool = False,
    ) -> None:
        losses = []
        previous_loss_total = 99999999999
        for epoch in range(n_epochs):
            for batch_num, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                self.zero_grad()
                output = self(data)

                output = output.to(torch.float32)
                target = target.to(torch.float32)

                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                train_loss = loss.item()
                losses.append(train_loss)

                # total = target.shape[0] * target.shape[1]
                # predicted = torch.where(output > 0.5, 1, 0)
                # train_correct = torch.sum(predicted == target)
                # if VERBOSE:
                #     print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                #           (epoch + 1, batch_num + 1, len(train_loader), train_loss,
                #            100. * train_correct / total, train_correct, total))

                plt.plot(losses)
            plt.show(block=False)

            with torch.no_grad():
                # dev_total = 0
                # dev_correct = 0
                loss_total = 0
                for dev_data, dev_target in dev_loader:
                    dev_data, dev_target \
                        = dev_data.to(device), dev_target.to(device)

                    dev_output = self(dev_data)

                    dev_output = dev_output.to(torch.float32)
                    dev_target = dev_target.to(torch.float32)

                    loss_total += loss_function(dev_output, dev_target).item()

                    # predicted = torch.where(dev_output > 0.5, 1, 0)

                    # dev_correct += torch.sum(predicted == dev_target)
                    # dev_total += dev_target.shape[0] * dev_target.shape[1]
                # print('Dev: Epoch %d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                #           (epoch + 1, loss_total,
                #            100. * dev_correct / dev_total, dev_correct, dev_total))

                # early stopping
                if previous_loss_total < loss_total:
                    break
                previous_loss_total = loss_total
