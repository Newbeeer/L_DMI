# fashion - data preparation and model for Fashion-MNIST

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import codecs
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float)
parser.add_argument('--s', type=int)
parser.add_argument('--c', type=int)
parser.add_argument('--device', type=int)
args = parser.parse_args()
torch.cuda.set_device(args.device)
r = args.r
root = './'
batch_size = 128
if args.c == 1: # class-independent noise
    conf_matrix = [[1 - r/2,r/2], [r/2, 1-r/2]]
if args.c == 2: # class-dependent noise (a)
    conf_matrix = [[1-r,r], [0, 1]]
if args.c == 3: # class-dependent noise (b)
    conf_matrix = [[1,0], [r, 1-r]]

class fashion(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train, valid, test, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.valid = valid
        self.test = test

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        train_data, train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))

        test_data, test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))

        self.train_data = train_data[0:50000]
        self.valid_data = train_data[50000:60000]
        self.test_data = test_data
        self.train_label = train_labels[0:50000]
        self.valid_label = train_labels[50000:60000]
        self.test_label = test_labels

        np.random.seed(args.s)

        for idx, target in enumerate(self.train_label):
            if self.train_label[idx] == 8:
                self.train_label[idx] = 0
            else:
                self.train_label[idx] = 1

        for idx, target in enumerate(self.valid_label):
            if self.valid_label[idx] == 8:
                self.valid_label[idx] = 0
            else:
                self.valid_label[idx] = 1

        for idx, target in enumerate(self.test_label):
            if self.test_label[idx] == 8:
                self.test_label[idx] = 0
            else:
                self.test_label[idx] = 1

        for idx, target in enumerate(self.train_label):
            target = int(np.random.choice(2, 1, p=np.array(conf_matrix[target])))
            self.train_label[idx] = target
        for idx, target in enumerate(self.valid_label):
            target = int(np.random.choice(2, 1, p=np.array(conf_matrix[target])))
            self.valid_label[idx] = target

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        elif self.valid:
            img, target = self.valid_data[index], self.valid_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.valid:
            return len(self.valid_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = fashion(root=root,
                        train=True,
                        valid=False,
                        test=False,
                        transform=transform_train,
                        )

valid_dataset = fashion(root=root,
                        train=False,
                        valid=True,
                        test=False,
                        transform=transform_test,
                        )

test_dataset = fashion(root=root,
                       train=False,
                       valid=False,
                       test=True,
                       transform=transform_test,
                       )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=16)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=16)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=16)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out
