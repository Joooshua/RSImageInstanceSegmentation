import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Alpha(1D Tensor, Variable): is a tensor which is the class loss weight related to frequency of the samples in the training dataset
# gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                               putting more focus on hard, misclassiﬁed examples
# size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                               However, if the field size_average is set to False, the losses are
#                               instead summed for each minibatch.
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target):
        # convert output to pseudo probability
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1-probs, self.gamma)

        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            focal_loss = (focal_loss/focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
        min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()
