# dataset - preparing data and adding label noise

from __future__ import print_function
import torch
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms

import argparse

# batch size
batch_size = 256

# r: noise amount s: random seed
parser = argparse.ArgumentParser()
parser.add_argument('--r',type=float)
parser.add_argument('--s',type=int)
parser.add_argument('--device',type=int)
parser.add_argument('--root', type=str, help='Path for loading CIFAR10 dataset')
parser.add_argument('--noise-type', type=str, default='class-dependent',
                        help='[class-dependent,class-independent]')
args = parser.parse_args()
torch.cuda.set_device(args.device)

# data root:
root = args.root
# noise:
r = args.r
conf_matrix = torch.eye(10)
# Uniform (Class-independent) noise:
if args.noise_type == 'class-independent':
    for i in range(10):
        for j in range(10):
            if i == j:
                conf_matrix[i][i] = 1 - r
            else:
                conf_matrix[i][j] = (r) / 9
else:
    # Class-dependent noise
    conf_matrix[9][1] = r
    conf_matrix[9][9] = 1 - r
    conf_matrix[2][0] = r
    conf_matrix[2][2] = 1 - r
    conf_matrix[4][7] = r
    conf_matrix[4][4] = 1 - r
    conf_matrix[3][5] = r
    conf_matrix[3][3] = 1 - r



class CIFAR10_(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ]

    valid_list = [
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, valid = False, test = False, noisy=False,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.valid = valid
        self.test = test
        self.noisy = noisy

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        if self.valid:
            downloaded_list = self.valid_list
        if self.test:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        np.random.seed(args.s)
        if noisy == True:
            for idx, target in enumerate(self.targets):
                target = int(np.random.choice(10, 1, p=np.array(conf_matrix[target])))
                self.targets[idx] = target

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = index

        return idx, img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset_noisy = CIFAR10_(root=root, train=True, valid=False, test=False, noisy=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
train_loader_noisy = torch.utils.data.DataLoader(dataset=train_dataset_noisy, batch_size=batch_size, shuffle=True, num_workers=16)
train_loader_noisy_unshuffle = torch.utils.data.DataLoader(dataset = train_dataset_noisy, batch_size=batch_size, shuffle =False)

valid_dataset_noisy = CIFAR10_(root=root, train=False, valid=True, test=False, noisy=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
valid_loader_noisy = torch.utils.data.DataLoader(dataset = valid_dataset_noisy, batch_size = batch_size, shuffle = False, num_workers=16)

test_dataset = CIFAR10_(root=root, train=False, valid=False, test=True, noisy=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
test_loader_ = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=16)
