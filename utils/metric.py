# encoding: utf-8

import numpy as np
import torch
from torchmetrics import JaccardIndex

np.seterr(divide='ignore', invalid='ignore')



def cal_mean_iou(pred, target):
    score = torch.exp(pred) # B, C, H, W
    jaccard = JaccardIndex(task="multiclass", num_classes=score.shape[1], ignore_index = 255).to(score.get_device())    
    
    mean_iou = jaccard(score, target)
    # print('mean iou: ',mean_iou)
    return mean_iou.detach().cpu().numpy()

# gt = 255
# pred.shape == gt.shape --> 1xHxW
def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)     # k --> gt.size--> indexes --> 0, 1
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))    # True False
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    # print(hist.shape)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:]) # useless for NYUDv2

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc