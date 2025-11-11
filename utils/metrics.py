import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

class Accuracy:
    def __init__(self, num_classes=11):
        self.qty = 0
        self.corrects = 0
        self.num_classes = num_classes

    def update(self, pred: Tensor, target: Tensor):
        pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
        pred = pred.transpose(1, 2)  # N,C,H*W => N,H*W,C
        _, pred = torch.max(pred, 2) # N,H*W,C => N,H*W

        target = target.view(target.size(0), -1)  # N,H,W => N,H*W

        idx = target < self.num_classes

        pred = pred[idx]
        target = target[idx]

        self.corrects += torch.sum(pred == target)
        self.qty += pred.size(0)

    def compute(self):
        return self.corrects/self.qty


class MIoU:
    def __init__(self, num_classes=11, ignore_class=None):

        if ignore_class is None:
            self.intersection = np.zeros(num_classes)
            self.union = np.zeros(num_classes)+ 1e-10
        else:
            self.intersection = np.zeros(num_classes-1)
            self.union = np.zeros(num_classes-1)+ 1e-10


        self.num_classes = num_classes
        self.ignore_class = ignore_class

    def update(self, _pred: Tensor, _target: Tensor, _weights: Tensor = None, _vol: Tensor = None):

        pred = _pred.clone().view(_pred.size(0), _pred.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        pred = pred.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        _, pred = torch.max(pred, 2) # N,H*W*D,C => N,H*W*D

        target = _target.clone().view(_target.size(0), -1) # N,H,W,D => N,H*W*D

        idx = (target != 255)
        target[target == 255] = 0


        if _weights is not None:
            weights = _weights.clone()
            weights = weights.view(target.size(0), -1)  # N,H,W => N,H*W
            idx = idx & (weights != 0.0)

        if self.ignore_class is None:
            idx = idx &(target < self.num_classes)
        else:
            idx = idx & (target < self.num_classes) & (target != self.ignore_class)

        if _vol is not None:
            vol = _vol.view(target.size(0), -1)  # N,H,W => N,H*W
            idx = idx & (torch.logical_or(torch.abs(vol)<1,vol==-1.0))

        _target = target[idx]
        _pred = pred[idx]

        c = -1
        for i in range(self.num_classes):
            if i != self.ignore_class:
                c += 1
                inter = torch.sum(_pred[_pred==i] == _target[_pred==i])
                union = _pred[_pred==i].size(0) + _target[_target==i].size(0) - inter
                self.intersection[c] += inter
                self.union[c] += union

    def compute(self):
        iou = self.intersection/self.union
        return np.mean(iou)

    def per_class_iou(self):
        iou = self.intersection/self.union
        return iou


class CompletionIoU:
    def __init__(self):
        self.bs = 0
        self.iou = 0.0
        self.precision = 0.0
        self.recall = 0.0

    def update(self, _pred: Tensor, _target: Tensor, _weights: Tensor = None, ):

        # iou
        pred = _pred.clone().view(_pred.size(0), _pred.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        target = _target.clone().view(_target.size(0), -1)  # N,H,W,D => N,H*W*D
        if _weights is not None:
            weights = _weights.clone().view(target.size(0), -1)  # N,H,W,D => N,H*W*D

        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        if _weights is not None:
            weights  = weights.cpu().detach().numpy()
        pred =np.argmax(pred, 1)  # N,C,H*W*D => N,H*W*D
        # print(np.unique(pred))
        pred[pred > 0] = 1
        # print(np.unique(pred))

        _bs= pred.shape[0]
        self.bs += _bs
        
        for bs in range(_bs):
            y_true = target[bs, :]  # GT
            y_pred = pred[bs, :]
            if _weights is not None:
                weights_empty = weights[bs, :]

                y_pred = y_pred[np.where(np.logical_and(weights_empty != 0, y_true != 255))]
                y_true = y_true[np.where(np.logical_and(weights_empty != 0, y_true != 255))]
            else:
                y_pred = y_pred[ y_true != 255]
                y_true = y_true[ y_true != 255]

            y_true[y_true>0]=1


            _p, _r, _, _ = precision_recall_fscore_support(y_true, y_pred,average='binary')  
            _iou = 1 / (1 / _p + 1 / _r - 1) if _p else 0  
            self.iou += _iou
            self.precision += _p
            self.recall += _r

    def compute(self):
        comp_iou = self.iou / self.bs
        precision = self.precision / self.bs
        recall = self.recall / self.bs
        return comp_iou, precision, recall
