import numpy as np
import torch

class SegmentationMetrics:
    def __init__(self, n_classes, device, ignore=None):
        self.n_classes = n_classes
        self.device = device
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor([n for n in range(self.n_classes) if n not in self.ignore]).long()
    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None
    def addbatch(self, preds, targets):
        preds_row = preds.reshape(-1)
        targets_row = targets.reshape(-1)
        indices = torch.stack([preds_row, targets_row], dim=0)
        if self.ones is None or self.last_scan_size != indices.shape[-1]:
            self.ones = torch.ones((indices.shape[-1]), device=self.device).long()
            self.last_scan_size = indices.shape[-1]
        self.confusion_matrix = self.confusion_matrix.index_put_(tuple(indices), self.ones, accumulate=True)
    def getstats(self):
        confusion_matrix = self.confusion_matrix.clone()
        confusion_matrix[self.ignore] = 0
        confusion_matrix[:, self.ignore] = 0
        true_pos = confusion_matrix.diag()
        false_pos = confusion_matrix.sum(dim=1) - true_pos
        false_neg = confusion_matrix.sum(dim=0) - true_pos
        return true_pos, false_pos, false_neg
    def getiou(self):
        true_pos, false_pos, false_neg = self.getstats()
        intersection = true_pos
        union = true_pos + false_pos + false_neg + 1e-15
        iou = intersection/union
        iou_mean = (intersection[self.include]/union[self.include]).mean()
        return iou_mean, iou
    def getacc(self):
        true_pos, false_pos, false_neg = self.getstats()
        total_truepos = true_pos.sum()
        total = true_pos[self.include].sum() + false_pos[self.include].sum() + 1e-15
        accuracy_mean = total_truepos/total
        return accuracy_mean