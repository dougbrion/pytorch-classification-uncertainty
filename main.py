import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt

from helpers import get_device, rotate_img, one_hot_embedding
from data import dataloaders, digit_one
from train import train_model
from test import rotating_image_classification
from losses import uncertainty_loss, relu_evidence
from lenet import LeNet


def main():
    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.savefig('testfig.jpg')

    n_epochs = 10
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    num_classes = 10

    network = LeNet()
    # criterion = nn.CrossEntropyLoss()
    criterion = uncertainty_loss
    # criterion = F.nll_loss
    optimizer = optim.Adam(network.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)

    device = get_device()
    network = network.to(device)

    model, metrics = train_model(network, dataloaders, num_classes, criterion,
                                 optimizer, scheduler=exp_lr_scheduler, num_epochs=50, device=device, uncertainty=True)

    # checkpoint = torch.load("./results/model.pt")
    checkpoint = torch.load("./results/uncertainty_model.pt")

    network.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    network.eval()

    rotating_image_classification(
        digit_one, "uncertainty.jpg", uncertainty=True)


if __name__ == "__main__":
    main()
