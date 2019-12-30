import sys
from lenet import LeNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


from matplotlib import pyplot as plt
from helpers import get_device, rotate_img, one_hot_embedding
from train import train_model
import numpy as np


data_train = MNIST('./data/mnist',
                   download=True,
                   train=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor(),
                       transforms.Normalize(
                           (0.1307,), (0.3081,))]))
data_val = MNIST('./data/mnist',
                 train=False,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize((28, 28)),
                     transforms.ToTensor(),
                     transforms.Normalize(
                         (0.1307,), (0.3081,))]))
dataloader_train = DataLoader(
    data_train, batch_size=256, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=1024, num_workers=8)
dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}


examples = enumerate(dataloader_val)
batch_idx, (example_data, example_targets) = next(examples)


digit_one, _ = data_val[5]

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

criterion = nn.CrossEntropyLoss()

network = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=2e-3)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
network = network.to(device)

# train_model(network, dataloaders, num_classes, criterion,
#             optimizer, num_epochs=3, device=device)

checkpoint = torch.load("./results/saved_model.pt")

network.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

network.eval()

# This method rotates an image counter-clockwise and classify it for different degress of rotation.
# It plots the highest classification probability along with the class label for each rotation degree.


def rotating_image_classification(img, uncertainty=None):
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

        nimg = np.clip(a=nimg, a_min=0, a_max=1)

        rimgs[:, i*28:(i+1)*28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        img_variable = img_variable.to(device)

        if uncertainty is None:
            output = network(img_variable)
            _, preds = torch.max(output, 1)
            prob = F.softmax(output, dim=1)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()

        else:
            output = network(img_variable)
            _, preds = torch.max(output, 1)
            prob = F.softmax(output, dim=1)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            lu.append(u.mean())
        one_hot = one_hot_embedding(preds[0])
        scores += one_hot.numpy()
        ldeg.append(deg)
        lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    print(labels)
    lp = np.array(lp)[:, labels]
    print(lp)
    c = ['black', 'blue', 'red', 'brown', 'purple', 'cyan']
    marker = ['s', '^', 'o']*2
    labels = labels.tolist()
    plt.figure(figsize=[6.2, 10])
    fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [6, 1]})

    for i in range(len(labels)):
        print("plotting")
        print(ldeg, lp[:, i])
        axs[0].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    # if uncertainty is not None:
    #     labels += ['uncertainty']
    #     plt.plot(ldeg, lu, marker='<', c='red')
    fig.subplots_adjust(hspace=.5)

    axs[0].legend(labels)

    axs[0].set_xlim([0, Mdeg])
    axs[0].set_ylim([0, 1])
    axs[0].set_xlabel('Rotation Degree')
    axs[0].set_ylabel('Classification Probability')

    axs[1].imshow(1 - rimgs, cmap='gray')
    axs[1].axis('off')
    plt.pause(0.001)

    plt.savefig("nums.jpg")


rotating_image_classification(digit_one)
