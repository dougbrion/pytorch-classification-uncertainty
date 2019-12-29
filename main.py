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
import numpy as np


image_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,), (0.3081,))])

data_train = MNIST('./data/mnist',
                   download=True,
                   train=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor(),
                       transforms.Normalize(
                           (0.1307,), (0.3081,))]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((28, 28)),
                      transforms.ToTensor(),
                      transforms.Normalize(
                          (0.1307,), (0.3081,))]))
dataloader_train = DataLoader(
    data_train, batch_size=256, shuffle=True, num_workers=8)
dataloader_test = DataLoader(data_test, batch_size=1024, num_workers=8)

network = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=2e-3)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

examples = enumerate(dataloader_test)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

digit_one, _ = data_test[5]
print(digit_one.shape)

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

# optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                       momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(dataloader_train.dataset) for i in range(n_epochs + 1)]
criterion = nn.CrossEntropyLoss()


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(dataloader_train):
        optimizer.zero_grad()
        output = network(data)
        # target = one_hot_embedding(target).long()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader_train.dataset),
                100. * batch_idx / len(dataloader_train), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(dataloader_train.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader_test:
            output = network(data)
            # target = one_hot_embedding(target).long()
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(dataloader_test.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader_test.dataset),
        100. * correct / len(dataloader_test.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()


network.load_state_dict(torch.load('./results/model.pth'))
optimizer.load_state_dict(torch.load('./results/optimizer.pth'))

network.eval()


# This method rotates an image counter-clockwise and classify it for different degress of rotation.
# It plots the highest classification probability along with the class label for each rotation degree.
def rotating_image_classification(img, threshold=0.5):
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)
        # print(nimg.shape)
        # plt.imsave('asdf.jpg', nimg)

        nimg = np.clip(a=nimg, a_min=0, a_max=1)

        rimgs[:, i*28:(i+1)*28] = nimg
        # print(rimgs.shape)
        # plt.imsave('asdff.jpg', rimgs)
        # nimg = np.insert(nimg, 0, 1)
        # nimg = nimg[np.newaxis, ...]
        # print(nimg.shape)
        # print(nimg.numpy().shape)
        # nimg = nimg.numpy()
        trans = transforms.ToTensor()
        # print(trans(nimg).shape)
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        output = network(img_variable)
        _, preds = torch.max(output, 1)
        prob = F.softmax(output, dim=1)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()

        # print(prob)
        print(preds)
        # if uncertainty is None:
        #     output = network(digit_one)
        # else:
        #     p_pred_t, u = sess.run([prob, uncertainty], feed_dict=feed_dict)
        #     lu.append(u.mean())
        one_hot = one_hot_embedding(preds[0])
        print(one_hot)
        scores += one_hot.numpy()
        print(scores)
        # print(scores)
        # print(scores)
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
    # fig.suptitle('Vertically stacked subplots')

    for i in range(len(labels)):
        print("plotting")
        print(ldeg, lp[:, i])
        axs[0].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    # if uncertainty is not None:
    #     labels += ['uncertainty']
    #     plt.plot(ldeg, lu, marker='<', c='red')

    axs[0].legend(labels)

    axs[0].set_xlim([0, Mdeg])
    axs[0].set_ylim([0, 1])
    axs[0].set_xlabel('Rotation Degree')
    axs[0].set_ylabel('Classification Probability')
    # plt.pause(0.001)
    # plt.show()
    print("HERE")
    # fig, ax = plt.subplots()

    axs[1].imshow(1 - rimgs, cmap='gray')
    axs[1].axis('off')
    plt.pause(0.001)
    # # plt.show()
    # plt.draw()
    plt.savefig("nums.jpg")


rotating_image_classification(digit_one)
