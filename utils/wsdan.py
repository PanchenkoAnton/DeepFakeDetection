import os
import json
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from .transform import augmentation, trans


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
                (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(
                size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def batch_augment(images, attention_map, mode='crop', theta=0.5,
                  padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()
    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()
            crop_mask = F.upsample_bilinear(atten_map,
                                            size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(
                int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH),
                0)
            height_max = min(
                int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH),
                imgH)
            width_min = max(
                int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW),
                0)
            width_max = min(
                int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW),
                imgW)
            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :,
                                    height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images
    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()
            drop_masks.append(
                F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images


class DfdcDataset(Dataset):

    def __init__(self, datapath="", phase='train', resize=(300, 300)):
        assert phase in ['train', 'val', 'test']
        if phase == 'val':
            phase = 'valid'
        self.phase = phase
        self.resize = resize
        self.num_classes = 2
        self.epoch = 0
        self.next_epoch()
        self.aug = augmentation
        self.trans = trans
        self.datapath = datapath

    def next_epoch(self):
        with open('dfdc.json') as f:
            dfdc = json.load(f)
        if self.phase == 'train':
            trainset = dfdc['train']+dfdc['valid']
            tr = list(filter(lambda x: x[1] == 0, trainset))
            tf = random.sample(list(filter(lambda x: x[1] == 1, trainset)), len(tr))
            self.dataset = tr+tf
        if self.phase == 'valid':
            validset = dfdc['test']
            tr = list(filter(lambda x: x[1] == 0, validset))
            tf = random.sample(list(filter(lambda x: x[1] == 1, validset)), len(tr))
            self.dataset = tr+tf
        if self.phase == 'test':
            self.dataset = dfdc['test']
        self.epoch += 1

    def __getitem__(self, item):
        try:
            vid = self.dataset[item // 20]
            ind = str(item % 20 * 12 + self.epoch % 12)
            ind = '0'*(3-len(ind))+ind+'.png'
            image = cv2.imread(os.path.join(self.datapath, vid[0], ind))
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               self.resize)
            if self.phase == 'train':
                image = self.aug(image=image)['image']
            return self.trans(image), vid[1]
        except:
            return self.__getitem__((item + 250) % (self.__len__()))

    def __len__(self):
        return len(self.dataset) * 20

