import torch
import torch.nn.functional as F
from helpers


def relu_evidence(y):
    return F.relu(y)


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def mse_loss():
    pass


def uncertainty_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    loss = torch.mean(mse_loss(target, alpha, epoch_num,
                               num_classes, annealing_step, device=device))
    l2_loss = F.mse_loss(output, target) * 0.005
    return loss + l2_loss, evidence
