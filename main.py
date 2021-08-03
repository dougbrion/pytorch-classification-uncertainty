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
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
from data import dataloaders, digit_one
from train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet


def main():

    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train", action="store_true", help="To train the network."
    )
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument(
        "--examples", action="store_true", help="To example MNIST data."
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--dropout", action="store_true", help="Whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", action="store_true", help="Use uncertainty or not."
    )
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    uncertainty_type_group.add_argument(
        "--digamma",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    uncertainty_type_group.add_argument(
        "--log",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )
    args = parser.parse_args()

    if args.examples:
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
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
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)

        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if use_uncertainty:
            if args.digamma:
                torch.save(state, "./results/model_uncertainty_digamma.pt")
                print("Saved: ./results/model_uncertainty_digamma.pt")
            if args.log:
                torch.save(state, "./results/model_uncertainty_log.pt")
                print("Saved: ./results/model_uncertainty_log.pt")
            if args.mse:
                torch.save(state, "./results/model_uncertainty_mse.pt")
                print("Saved: ./results/model_uncertainty_mse.pt")

        else:
            torch.save(state, "./results/model.pt")
            print("Saved: ./results/model.pt")

    elif args.test:

        use_uncertainty = args.uncertainty
        device = get_device()
        model = LeNet()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())

        if use_uncertainty:
            if args.digamma:
                checkpoint = torch.load("./results/model_uncertainty_digamma.pt")
                filename = "./results/rotate_uncertainty_digamma.jpg"
            if args.log:
                checkpoint = torch.load("./results/model_uncertainty_log.pt")
                filename = "./results/rotate_uncertainty_log.jpg"
            if args.mse:
                checkpoint = torch.load("./results/model_uncertainty_mse.pt")
                filename = "./results/rotate_uncertainty_mse.jpg"

        else:
            checkpoint = torch.load("./results/model.pt")
            filename = "./results/rotate.jpg"

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()

        rotating_image_classification(
            model, digit_one, filename, uncertainty=use_uncertainty
        )

        test_single_image(model, "./data/one.jpg", uncertainty=use_uncertainty)
        test_single_image(model, "./data/yoda.jpg", uncertainty=use_uncertainty)


if __name__ == "__main__":
    main()
