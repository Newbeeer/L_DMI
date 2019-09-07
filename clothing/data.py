"""
data - loading data from Clothing1M
"""
import numpy as np
import torch
import torch.utils
from PIL import Image
import os
import torchvision.transforms as transforms


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, img_transform, train, valid, test):
        self.root = root
        if train==True:
            flist = os.path.join(root, "annotations/noisy_train.txt")
        if valid==True:
            flist = os.path.join(root, "annotations/clean_val.txt")
        if test==True:
            flist = os.path.join(root, "annotations/clean_test.txt")

        self.imlist = self.flist_reader(flist)
        self.transform = img_transform
        self.train = train

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
                row = line.split(" ")
                impath =  self.root + row[0]
                imlabel = row[1]
                imlist.append((impath, int(imlabel)))
        return imlist


train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])

