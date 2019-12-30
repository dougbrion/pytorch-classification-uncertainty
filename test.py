import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from losses import relu_evidence
from helpers import rotate_img, one_hot_embedding

# This method rotates an image counter-clockwise and classify it for different degress of rotation.
# It plots the highest classification probability along with the class label for each rotation degree.


def rotating_image_classification(img, filename, uncertainty=None, threshold=0.5):
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
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            print(uncertainty)
            _, preds = torch.max(output, 1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            print(prob)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()

            lu.append(uncertainty.mean())
        # one_hot = one_hot_embedding(preds[0])
        # scores += one_hot.numpy()
        print(prob)
        scores += prob.detach().cpu().numpy() >= threshold
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

    if uncertainty is not None:
        labels += ['uncertainty']
        axs[0].plot(ldeg, lu, marker='<', c='red')

    fig.subplots_adjust(hspace=.5)

    axs[0].legend(labels)

    axs[0].set_xlim([0, Mdeg])
    axs[0].set_ylim([0, 1])
    axs[0].set_xlabel('Rotation Degree')
    axs[0].set_ylabel('Classification Probability')

    axs[1].imshow(1 - rimgs, cmap='gray')
    axs[1].axis('off')
    plt.pause(0.001)

    plt.savefig(filename)
