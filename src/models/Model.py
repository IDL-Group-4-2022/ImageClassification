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
    ) -> None:
        train_losses = []
        dev_losses = []
        dev_batches = []
        dev_loss_lower_than_min_consecutive_count = 0
        #plt.ion()
        #figure, axs = plt.subplots(1, 2)
        #train_line, = axs[0].plot(range(10), range(10), label='train_loss')
        #axs[0].set_title('Train Loss')
        #dev_line, = axs[1].plot(dev_batches, dev_losses, label='dev_loss')
        #axs[1].set_title('Development Loss')
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
                        f'/{len(train_loader)}: Loss: {train_loss:.4f}'
                    )
                if verbose > 1:
                    predicted = torch.where(output > 0.5, 1, 0)
                    print_metrics_multilabel(
                        target.cpu().numpy(), predicted.cpu().numpy()
                    )

                #train_line.set_xdata(range(len(train_losses)))
                #train_line.set_ydata(train_losses)
                #axs[0].set_xlim(0, len(train_losses))
                #axs[0].set_ylim(0, max(train_losses) * 1.2)
                #figure.canvas.draw()
                #figure.canvas.flush_events()

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

                dev_losses.append(loss_total.item())
                dev_batches.append(len(train_losses))
                #axs[1].set_xlim(0, dev_batches[-1] * 1.2)
                #axs[1].set_ylim(0, max(dev_losses) * 1.2)
                #dev_line.set_xdata(dev_batches)
                #dev_line.set_ydata(dev_losses)
                #figure.canvas.draw()
                #figure.canvas.flush_events()

                if verbose > 0:
                    print(
                        f'Development: Epoch {epoch + 1}'
                        f': Loss: {loss_total:.4f}'
                    )
                    predicted = torch.where(dev_output > 0.5, 1, 0)
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
        #plt.ioff()
        #plt.show(block=True)
        #Development: Epoch 30: Loss: 9.6065
        #Macro Precision: 0.4025974025974026
        #Micro Precision: 0.7049180327868853
        #Samples Precision: 0.6064814814814814
        #Macro Recall: 0.2800865800865801
        #Micro Recall: 0.6142857142857143
        #Samples Recall: 0.5185185185185185
        #Macro F1: 0.314787784679089
        #Micro F1: 0.6564885496183206
        #Samples F1: 0.5272486772486773
