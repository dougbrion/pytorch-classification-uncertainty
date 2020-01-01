import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import argparse
from matplotlib import pyplot as plt

from helpers import get_device, rotate_img, one_hot_embedding
from data import dataloaders, digit_one
from train import train_model
from test import rotating_image_classification
from losses import uncertainty_loss, relu_evidence
from lenet import LeNet


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="To train the network.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--dropout", action="store_true",
                        help="Use dropout or not")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Use uncertainty or not")
    parser.add_argument("--test", action="store_true",
                        help="To test the network.")
    parser.add_argument("--examples", action="store_true",
                        help="To show data.")
    args = parser.parse_args()

    if args.examples:
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")

    elif args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = 10

        model = LeNet(dropout=args.dropout)

        if use_uncertainty:
            criterion = uncertainty_loss
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters())

        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)

        _, _ = train_model(model, dataloaders, num_classes, criterion,
                           optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs,
                           device=device, uncertainty=use_uncertainty)

    elif args.test:
        use_uncertainty = args.uncertainty

        model = LeNet()
        optimizer = optim.Adam(model.parameters())

        if use_uncertainty:
            checkpoint = torch.load("./results/uncertainty_model.pt")
            filename = "./results/uncertainty_rotate.jpg"
        else:
            checkpoint = torch.load("./results/model.pt")
            filename = "./results/rotate.jpg"

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()

        rotating_image_classification(
            model, digit_one, filename, uncertainty=use_uncertainty)

        # n_epochs = 10
        # learning_rate = 0.01
        # momentum = 0.5
        # log_interval = 10
        # num_classes = 10

        # # criterion = nn.CrossEntropyLoss()
        # criterion = uncertainty_loss
        # # criterion = F.nll_loss
        # optimizer = optim.Adam(network.parameters())
        # exp_lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, step_size=7, gamma=0.1)

        # device = get_device()
        # network = network.to(device)

        # model, metrics = train_model(network, dataloaders, num_classes, criterion,
        #                              optimizer, scheduler=exp_lr_scheduler, num_epochs=50, device=device, uncertainty=True)

        # # checkpoint = torch.load("./results/model.pt")


if __name__ == "__main__":
    main()
