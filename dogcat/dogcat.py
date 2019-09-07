# dogcat - data preparation and model for Dogs vs. Cats

import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float)
parser.add_argument('--s', type=int)
args = parser.parse_args()
seed = args.s

data_root = '/data1/caopeng/dogdata/'
batch_size = 128
num_classes = 2

train_transform = transforms.Compose([
            transforms.Resize((150, 150),interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
test_transform = transforms.Compose([
            transforms.Resize((150, 150),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])

conf_matrix = [[1, 0], [args.r, 1-args.r]]

class Image_(torch.utils.data.Dataset):
    def __init__(self, root, img_transform, train, val, test):
        self.root = root
        if train :
            flist = os.path.join(root, "train_file.csv")
        elif val :
            flist = os.path.join(root, "val_file.csv")
        else:
            flist = os.path.join(root, "test_file.csv")
        self.imlist = self.flist_reader(flist)
        self.transform = img_transform
        self.train = train
        self.val = val
        self.test = test
        np.random.seed(seed)
        if self.train or self.val:
            for idx, (impath, target) in enumerate(self.imlist):
                target = int(np.random.choice(num_classes, 1, p = np.array(conf_matrix[target])))
                self.imlist[idx] = (impath, target)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return index, img, target

    def __len__(self):
        return len(self.imlist)

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(",")
                impath =  data_root + row[0]
                imlabel = row[1]
                imlist.append((impath, int(imlabel)))
        return imlist

train_dataset = Image_(root=data_root, img_transform=train_transform, train=True, val=False, test=False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=16)
val_dataset = Image_(root=data_root, img_transform=test_transform, train=False, val=True, test=False)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers=16)
test_dataset = Image_(root=data_root, img_transform=test_transform, train=False, val=False, test=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=16)
train_loader_unshuffle = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle =False, num_workers=16)

class VGG(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self, vgg_name):
        super(VGG, self).__init__()

        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.features = self._make_layers(self.cfg[vgg_name])
        self.classifier = nn.Linear(8192, num_classes)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out,dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
