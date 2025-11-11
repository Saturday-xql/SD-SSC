import torch.nn as nn
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_class_weights(dataloader, num_classes, dev):
    # get class frequencies
    freq = np.zeros((num_classes,))
    for _, labels in dataloader:
        for c in range(num_classes):
            freq[c] += torch.sum(labels == c)
    class_probs = freq / np.sum(freq)

    class_weights = np.append((1/class_probs)/np.sum(1/class_probs),0)

    print(class_weights)
    print(sum(class_weights))
    return torch.from_numpy(class_weights.astype(np.float32)).to(dev)


class CustomCrossEntropy(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Tensor = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', num_classes=40) -> None:
        super(CustomCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.weight is not None:
            self.weight = self.weight.to(input.device)
        #print(input.shape)
        #print(target.shape)

        input = input.view(input.size(0), input.size(1), -1).contiguous()  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C

        target = target.view(target.size(0), -1).contiguous()  # N,H,W => N,H*W

        #print(input.shape)
        #print(target.shape)

        idx = target < self.num_classes

        input = input[idx]
        target = target[idx]

        return super(CustomCrossEntropy, self).forward(input, target)


class MscCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', num_classes=11):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, preds, target):
        loss = 0
        for item in preds:
            print('item.shape:',item.shape)
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='nearest').long()
           
            item = item.view(item.size(0), item.size(1), -1).contiguous()  # N,C,H,W => N,C,H*W
            item = item.transpose(1, 2)  # N,C,H*W => N,H*W,C

            item_target = item_target.view(item_target.size(0), -1).contiguous()  # N,H,W => N,H*W
            # print(item_target)
            # print(torch.unique(item_target))
            # print(self.num_classes)
            idx = item_target < self.num_classes
            item = item[idx]
            item_target = item_target[idx]
            # print(torch.unique(item))
            # print(item.shape)
            loss += F.cross_entropy(item, item_target, weight=self.weight, reduction=self.reduction)

        return loss / len(preds)


class SSCCrossEntropy(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Tensor = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', num_classes=2) -> None:
        super(SSCCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        #print(input.shape)
        #print(target.shape)

        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C

        target = target.view(target.size(0), -1)  # N,H,W => N,H*W
        weights = weights.view(weights.size(0), -1)  # N,H,W => N,H*W

        #print(input.shape)
        #print(target.shape)

        epsilon = 1e-10

        occluded = torch.logical_and(~torch.abs(weights).eq(1.0), ~weights.eq(0))

        occupied = torch.abs(weights).eq(1.0)

        #ratio = (4. * torch.sum(occupied)) / (torch.sum(occluded) + epsilon)
        ratio = (2. * torch.sum(occupied)) / (torch.sum(occluded) + epsilon)
        #ratio = (1. * torch.sum(occupied)) / (torch.sum(occluded) + epsilon)

        rand = torch.rand(size=list(target.size())).to(weights.device)

        idx = torch.logical_or(occupied, torch.logical_and(occluded, rand <= ratio))

        input = input[idx]
        target = target[idx]

        return super(SSCCrossEntropy, self).forward(input, target.type(torch.LongTensor).to(weights.device))


class BCELoss(nn.Module):
    def __init__(self, weight: Tensor = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    def forward(self, logits, labels, weights):
        """
        Args:
            input: (∗), where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        logits = logits.view(labels.size(0), -1)
        labels = labels.view(labels.size(0), -1)  # N,H,W => N,H*W
        weights = weights.view(weights.size(0), -1)  # N,H,W => N,H*W

        idx = weights != 0.0

        logits = logits[idx]
        labels = labels[idx].float()
        #print(logits.shape)
        #print(labels.shape)
        
        loss = self.bce(logits, labels)
       
        return loss


class WeightedSSCCrossEntropy(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Tensor = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', num_classes=12) -> None:
        super(WeightedSSCCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.num_classes = num_classes
        self.ignore_index=ignore_index

    def forward(self, logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
        if self.weight is not None:
            self.weight = self.weight.to(logits.device)
        #print(input.shape)
        #print(target.shape)
        logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
        logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C

        labels = labels.view(labels.size(0), -1)  # N,H,W => N,H*W
        weights = weights.view(weights.size(0), -1)  # N,H,W => N,H*W

        idx_l=labels!=self.ignore_index
        idx_w = weights != 0.0
        idx=idx_l & idx_w

        logits = logits[idx]
        labels = labels[idx]

        return super(WeightedSSCCrossEntropy, self).forward(logits, labels.type(torch.LongTensor).to(weights.device))


class weightedssccrossentropy(nn.Module):
    def __init__(self,ignore_index=-1,class_weight=None):
        super(weightedssccrossentropy,self).__init__()
        self.ignore_index = ignore_index
        self.class_weight=class_weight

        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weight)
    def forward(self, logits, target,weight=None):

        B, C, W, H, D = logits.shape
        logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target = target.reshape(-1)
        if weight is not None:
            weight = weight.reshape(-1)

            valid_mask = (weight != 0) & (target != self.ignore_index)
            logits = logits[ valid_mask,:]
            target = target[valid_mask]
        else:

            valid_mask = (target != self.ignore_index)
            logits = logits[valid_mask,:]
            target = target[valid_mask]


        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(logits.device)
        loss=self.cross_entropy(logits,target.long())
        return loss






