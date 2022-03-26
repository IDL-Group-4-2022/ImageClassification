"""
Parent class for our models which contains the training
"""

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from utils.metrics import print_metrics_multilabel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def do_train(
        self,
        n_epochs: int,
        device: str,
        loss_function: torch.nn.Module,
        optimizer: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        dev_loader: torch.utils.data.DataLoader,
        verbose: int = 1,
        interactive=False,
    ) -> None:
        """Train the classifier

        Args:
            n_epochs (int): number of epochs
            device (str): device
            loss_function (torch.nn.Module): loss function
            optimizer (torch.nn.Module): optimizer
            train_loader (torch.utils.data.DataLoader): training dataloader
            dev_loader (torch.utils.data.DataLoader): development dataloader
            verbose (int, optional): if 0, do not print accuracies. If > 0, print loss, if > 1, print loss and other performance metrics. Defaults to 1.
            interactive (bool, optional): if True, plot loss. Defaults to False.
        """
        train_losses = []
        dev_losses = []
        dev_batches = []
        dev_loss_lower_than_min_consecutive_count = 0
        if interactive:
            plt.ion()
            figure, axs = plt.subplots(1, 2)
            train_line, = axs[0].plot(range(10), range(10), label='train_loss')
            axs[0].set_title('Train Loss')
            dev_line, = axs[1].plot(dev_batches, dev_losses, label='dev_loss')
            axs[1].set_title('Development Loss')
        min_loss = float('inf')
        for epoch in range(n_epochs):
            self.train()  # set model to training mode
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
                train_losses.append(train_loss)

                if verbose > 0:
                    print(
                        f'Training: Epoch {epoch}/{n_epochs}'
                        f' - Batch {batch_num + 1}'
                        f'/{len(train_loader)}, Average Loss: {train_loss:.4f}',
                        flush=True
                    )
                if verbose > 1:
                    predicted = torch.where(torch.sigmoid(output) > 0.5, 1, 0)
                    print_metrics_multilabel(
                        target.cpu().numpy(), predicted.cpu().numpy()
                    )

                if interactive:
                    train_line.set_xdata(range(len(train_losses)))
                    train_line.set_ydata(train_losses)
                    axs[0].set_xlim(0, len(train_losses))
                    axs[0].set_ylim(0, max(train_losses) * 1.2)
                    figure.canvas.draw()
                    figure.canvas.flush_events()

            with torch.no_grad():
                self.eval()  # set model to eval mode
                loss_total = 0
                for dev_data, dev_target in dev_loader:
                    dev_data, dev_target \
                        = dev_data.to(device), dev_target.to(device)

                    dev_output = self(dev_data)

                    dev_output = dev_output.to(torch.float32)
                    dev_target = dev_target.to(torch.float32)

                    loss_total += loss_function(dev_output, dev_target)

                loss_total = loss_total.item() / len(dev_loader)
                dev_losses.append(loss_total)
                dev_batches.append(len(train_losses))
                if interactive:
                    axs[1].set_xlim(0, dev_batches[-1] * 1.2)
                    axs[1].set_ylim(0, max(dev_losses) * 1.2)
                    dev_line.set_xdata(dev_batches)
                    dev_line.set_ydata(dev_losses)
                    figure.canvas.draw()
                    figure.canvas.flush_events()

                if verbose > 0:
                    print(
                        f'Development: Epoch {epoch + 1}'
                        f', Average Loss: {loss_total:.4f}'
                    )
                    predicted = torch.where(torch.sigmoid(dev_output) > 0.5, 1, 0)
                    print_metrics_multilabel(
                        dev_target.cpu().numpy(), predicted.cpu().numpy()
                    )

                # early stopping
                if loss_total < min_loss:
                    min_loss = loss_total
                    dev_loss_lower_than_min_consecutive_count = 0
                elif min_loss < loss_total:
                    dev_loss_lower_than_min_consecutive_count += 1
                    if dev_loss_lower_than_min_consecutive_count == 3:
                        break

        if interactive:
            plt.ioff()
        else:
            figure, axs = plt.subplots(1, 2)
            top = max(max(train_losses), max(dev_losses)) * 1.2
            axs[0].plot(range(len(train_losses)), train_losses, label='train_loss')
            axs[0].set_title('Train Loss')
            axs[0].set_ylim(bottom=0, top=top)
            axs[1].plot(dev_batches, dev_losses, label='dev_loss')
            axs[1].set_title('Development Loss')
            axs[1].set_ylim(bottom=0, top=top)
        # plt.savefig('resources/models/Transferred.png')
        plt.savefig('resources/models/Cnn2.png')
