import os
import sys
import numpy as np
import math
import time
import shutil, errno
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


#### logging ####
def save_checkpoint(state, is_best, results_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(results_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(results_dir, filename),
            os.path.join(results_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_features(model, dataloader, max_images=10 ** 10, verbose=False):
    features, labels = [], []
    total = 0
    
    model.eval()
    
    for index, (img, label) in enumerate(dataloader):

        if total > max_images:
            break

        img, label = img.cuda(), label.cuda()

        features += list(model(img).data.cpu().numpy())
        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)

        total += len(img)

    return np.array(features), np.array(labels)


def baseeval(model, device, val_loader, criterion, args, epoch=0):
    """
    Evaluating on validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return top1.avg, top5.avg


def knn(model, device, val_loader, criterion, args, writer, epoch=0):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            # compute output
            output = F.normalize(model(images), dim=-1).data.cpu()
            features.append(output)
            labels.append(target)

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


#### OOD detection ####
def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)


def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=False):
    # Todo: calculate OOD scores for test (In-Distribution) & OOD data
    score_test=None
    score_ood=None
    raise NotImplementedError
    return score_test, score_ood


#### Dataloaders ####
def readloader(dataloader):
    images = []
    labels = []
    for img, label in dataloader:
        images.append(img)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def unnormalize(x, norm_layer):
    m, s = (
        torch.tensor(norm_layer.mean).view(1, 3, 1, 1),
        torch.tensor(norm_layer.std).view(1, 3, 1, 1),
    )
    return x * s + m