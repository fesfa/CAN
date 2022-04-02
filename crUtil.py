import numpy as np
import torch
import torch.nn as nn


def crLoss(data, label, class_num):
    data = data.cuda()
    label = label.cuda()
    groups = []
    group_center = []
    for i in range(class_num):
        temp1 = data[[label == i]]
        length = len(temp1)
        if length > 0:
            temp2 = sum(temp1)
            group_center.append(temp2 / length)
        groups.append(temp1)
    d1 = -mutiVertDistance(group_center) 
    d2 = 0.0
    for group in groups:
        if len(group) > 1:
            d2 += mutiVertDistance(group)
    d2 = d2 / len(groups)
    return d1, d2


def mutiVertDistance(arr):
    n = len(arr)
    temp = torch.zeros(n, arr[0].shape[0]).cuda()
    for i in range(n):
        temp[i] = arr[i]
    temp = temp / (temp ** 2).sum(axis=1, keepdims=True) ** 0.5
    distance = (torch.sum(1.0 - torch.mm(temp, temp.T)))
    return distance / n / (n - 1)


# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss


def new_contras(features, att):
    feat_sim = features/(features**2).sum(axis=1, keepdim=True)**0.5
    feat_sim = torch.matmul(feat_sim, feat_sim.T)
    att_sim = att/(att**2).sum(axis=1, keepdim=True)**0.5
    att_sim = torch.matmul(att_sim, att_sim.T)
    loss = feat_sim - att_sim
    loss = torch.sum(loss)
    return loss


# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear2(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear2, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss


def new_contras2(features, att, labels, d=0, t=0.1):
    feat_sim = features/(features**2).sum(axis=1, keepdim=True)**0.5
    feat_sim = torch.matmul(feat_sim, feat_sim.T)
    att_sim = att/(att**2).sum(axis=1, keepdim=True)**0.5
    att_sim = torch.matmul(att_sim, att_sim.T)

    loss = torch.exp((feat_sim - att_sim - d)**2/t)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().cuda()
    mask = 1 - mask
    loss = loss*mask

    loss = torch.sum(loss)/features.shape[0]
    return loss