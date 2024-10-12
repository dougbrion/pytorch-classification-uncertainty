import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from losses import relu_evidence
from helpers import rotate_img, one_hot_embedding, get_device

#整体思路是加载一张图像，预处理后输入到深度学习模型中进行分类。根据是否启用不确定性分析，
#代码会计算并输出预测结果、类别概率和不确定性值，并通过Matplotlib可视化结果。
def test_single_image(model, img_path, uncertainty=False, device=None):
    img = Image.open(img_path).convert("L")#打开图像文件，并将其转换为灰度模式（"L"表示亮度）。
    if not device:
        device = get_device()
    num_classes = 10
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    # transforms.Compose用于将多个图像转换操作组合在一起。它允许用户按顺序应用一系列转换，以便在处理图像时简化代码。
    #图像转换：定义一个图像转换序列：transforms.Resize((28, 28))：将图像调整为28x28像素。transforms.ToTensor()：将图像转换为PyTorch张量。
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)#在张量的第一个维度上增加一个维度，以便将其视为一个批次（batch）
    img_variable = Variable(img_tensor)#将张量封装为 Variable，以便进行自动求导
    img_variable = img_variable.to(device)

    if uncertainty:
        output = model(img_variable)#把处理好的图像数据传入模型得到输出
        evidence = relu_evidence(output)
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)#计算类别概率：通过α值计算每个类别的概率。
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()#展平张量：将输出、概率和预测类别展平为一维张量，以便后续处理。
        print("Predict:", preds[0])
        print("Probs:", prob)
        print("Uncertainty:", uncertainty)

    else:

        output = model(img_variable)
        _, preds = torch.max(output, 1)#同样使用 torch.max 获取预测类别
        prob = F.softmax(output, dim=1)#使用 F.softmax 计算每个类别的概率。
        output = output.flatten()#展平一维
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)
    #作图
    labels = np.arange(10)
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))

    axs[0].set_title("One")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")

    fig.tight_layout()

    plt.savefig("./results/{}".format(os.path.basename(img_path)))


def rotating_image_classification(
    model, img, filename, uncertainty=False, threshold=0.5, device=None#threshold：阈值，默认为 0.5，可能用于过滤分类结果
):
    if not device:
        device = get_device()
    num_classes = 10#定义了分类的类别数量为 10。
    Mdeg = 180#定义用于旋转的最大度数为 180。
    Ndeg = int(Mdeg / 10) + 1#计算每次旋转的角度，每 10 度一个，共会有 19 个实验
    ldeg = []#存储旋转角度。
    lp = []#存储分类概率
    lu = []#存储不确定性值
    classifications = []#储存最终的分类结果
    #初始化两个用于存储分类分数和图像数据的数组
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
    #使用 np.linspace 产生从 0 到 Mdeg 的 Ndeg 个均匀间隔的角度值。enumerate 会返回每个角度的索引 i 和角度 deg
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

        nimg = np.clip(a=nimg, a_min=0, a_max=1)

        rimgs[:, i * 28 : (i + 1) * 28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        img_variable = img_variable.to(device)

        if uncertainty:
            output = model(img_variable)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            _, preds = torch.max(output, 1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())
            lu.append(uncertainty.mean())

        else:

            output = model(img_variable)
            _, preds = torch.max(output, 1)
            prob = F.softmax(output, dim=1)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())

        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"] * 2
    labels = labels.tolist()
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})

    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if uncertainty:
        labels += ["uncertainty"]
        axs[2].plot(ldeg, lu, marker="<", c="red")

    print(classifications)

    axs[0].set_title('Rotated "1" Digit Classifications')
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")
    plt.pause(0.001)

    empty_lst = []
    empty_lst.append(classifications)
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")

    axs[2].legend(labels)
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")

    plt.savefig(filename)
