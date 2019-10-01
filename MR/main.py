import argparse
from os.path import dirname, abspath, join, exists
import os

import torch
from torch.optim import Adadelta, Adam, lr_scheduler
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from download_dataset import DATASETS
from preprocessors import DATASET_TO_PREPROCESSOR
import dictionaries
from dataloaders import TextDataset, TextDataLoader
from models.WordCNN import WordCNN

import utils

# Random seed
np.random.seed(0)
torch.manual_seed(0)

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('--dataset', type=str, default='MR', choices=DATASETS)
parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--initial_lr', type=float, default=0.01)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--r', type=float)
parser.add_argument('--seed', type=int)
subparsers = parser.add_subparsers(help='NLP Model')

## WordCNN
WordCNN_parser = subparsers.add_parser('WordCNN')
# WordCNN_parser.set_defaults(preprocess_level='word')
WordCNN_parser.add_argument('--preprocess_level', type=str, default='word', choices=['word', 'char'])
WordCNN_parser.add_argument('--dictionary', type=str, default='WordDictionary', choices=['WordDictionary', 'AllCharDictionary'])
WordCNN_parser.add_argument('--max_vocab_size', type=int, default=50000)
WordCNN_parser.add_argument('--min_count', type=int, default=None)
WordCNN_parser.add_argument('--start_end_tokens', type=bool, default=False)
group = WordCNN_parser.add_mutually_exclusive_group()
group.add_argument('--vector_size', type=int, default=128, help='Only for rand mode')
group.add_argument('--wordvec_mode', type=str, default=None, choices=['word2vec', 'glove'])
WordCNN_parser.add_argument('--min_length', type=int, default=5)
WordCNN_parser.add_argument('--max_length', type=int, default=300)
WordCNN_parser.add_argument('--sort_dataset', action='store_true')
WordCNN_parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'static', 'non-static', 'multichannel'])
WordCNN_parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
WordCNN_parser.add_argument('--epochs', type=int, default=50)
WordCNN_parser.set_defaults(model=WordCNN)

args = parser.parse_args()

seed = args.seed
r = args.r
conf_matrix = [[1,0], [r, 1-r]]
# conf_matrix = [[1-r,r], [0, 1]]
# conf_matrix = [[1 - r / 2, r / 2], [r / 2, 1 - r / 2]]

batch_size = args.batch_size

# Logging
model_name = args.model.__name__
logger = utils.get_logger(model_name)

# logger.info('Arguments: {}'.format(args))

# logger.info("Preprocessing...")
Preprocessor = DATASET_TO_PREPROCESSOR[args.dataset]
preprocessor = Preprocessor(args.dataset)
train_data, val_data, test_data = preprocessor.preprocess(level=args.preprocess_level)

# logger.info("Building dictionary...")
Dictionary = getattr(dictionaries, args.dictionary)
dictionary = Dictionary(args)
dictionary.build_dictionary(train_data)

# logger.info("Making dataset & dataloader...")
train_dataset = TextDataset(texts=train_data, dictionary=dictionary, conf_matrix=conf_matrix, seed=seed, train=True, sort=args.sort_dataset, min_length=args.min_length, max_length=args.max_length)
print("train: ", train_dataset.__len__())
train_dataloader = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=args.batch_size, shuffle=True)
train_dataloader_unshuffle = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=args.batch_size, shuffle=False)

val_dataset = TextDataset(texts=val_data, dictionary=dictionary, conf_matrix=conf_matrix, seed=seed, val=True, sort=args.sort_dataset, min_length=args.min_length, max_length=args.max_length)
print("val: ", val_dataset.__len__())
val_dataloader = TextDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=args.batch_size)

test_dataset = TextDataset(texts=test_data, dictionary=dictionary, conf_matrix=conf_matrix, seed=seed, test=True, sort=args.sort_dataset, min_length=args.min_length, max_length=args.max_length)
print("test: ", test_dataset.__len__())
test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=args.batch_size)

# logger.info("Training...")
CE_loss = nn.CrossEntropyLoss(size_average=False)
num_classes=2

def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = torch.matmul(y_onehot, outputs)
    loss = -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
    return loss

q = 0.7

def GCE_loss(outputs, target):
    # loss = (1 - h_j(x)^q) / q
    loss = 0
    outputs = F.softmax(outputs, dim=1)

    for i in range(outputs.size(0)):
        loss += (1.0 - (outputs[i][target[i]]) ** q) / q

    loss = loss / batch_size

    return loss

def train(train_loader, model, optimizer, criterion, trans_mat=None):
    model.train()

    for i, (idx, input, target) in enumerate(tqdm(train_loader)):
        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        if trans_mat is None:
            loss = criterion(output, target)
        else:
            loss = criterion(output, target, trans_mat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(valid_loader, model, criterion, trans_mat = None):

    model.eval()
    loss_t = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(tqdm(valid_loader)):
            if i == 1:
                break
            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            output = model(input)
            if trans_mat is None:
                loss = criterion(output, target)
            else:
                loss = criterion(output, target, trans_mat)
            loss_t += loss

    print('valid_loss=', loss_t.item())
    return loss_t

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(tqdm(test_loader)):
            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            total += target.size(0)
            output = model(input)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy

def main_ce():
    model = args.model(n_classes=preprocessor.n_classes, dictionary=dictionary, args=args).cuda()
    best_acc = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(args.epochs):
        print("epoch=", epoch)
        print('r=', args.r)
        print("traning model_ce...")
        learning_rate=0.001
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        train(train_loader=train_dataloader, model=model, optimizer=optimizer, criterion=CE_loss)
        print("validating model_ce...")
        valid_acc = test(model=model, test_loader=val_dataloader)
        print('valid_acc=', valid_acc)
        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(model, './model_ce_' + str(args.r) + '_' + str(args.seed))
            print("saved.")

def main_dmi():
    model = torch.load('./model_ce_' + str(args.r) + '_' + str(args.seed))
    best_valid_loss = validate(valid_loader=val_dataloader, model=model, criterion=DMI_loss)
    print('valid_loss=', best_valid_loss)
    torch.save(model, './model_dmi_' + str(args.r) + '_' + str(args.seed))
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(args.epochs):
        print("epoch=", epoch)
        print('r=', args.r)
        print("traning model_dmi...")
        learning_rate=0.0001
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        train(train_loader=train_dataloader, model=model, optimizer=optimizer, criterion=DMI_loss)
        print("validating model_dmi...")
        valid_loss = validate(valid_loader=val_dataloader, model=model, criterion=DMI_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, './model_dmi_' + str(args.r) + '_' + str(args.seed))
            print("saved.")

def main_gce():
    model = torch.load('./model_ce_' + str(args.r) + '_' + str(args.seed))
    best_valid_loss = validate(valid_loader=val_dataloader, model=model, criterion=GCE_loss)
    print('valid_loss=', best_valid_loss)
    torch.save(model, './model_gce_' + str(args.r) + '_' + str(args.seed))
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(args.epochs):
        print("epoch=", epoch)
        print('r=', args.r)
        print("traning model_gce...")
        learning_rate=0.0001
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        train(train_loader=train_dataloader, model=model, optimizer=optimizer, criterion=GCE_loss)
        print("validating model_gce...")
        valid_loss = validate(valid_loader=val_dataloader, model=model, criterion=GCE_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, './model_gce_' + str(args.r) + '_' + str(args.seed))
            print("saved.")

def find_trans_mat(model_ce):
    # estimate each component of matrix T based on training with noisy labels

    print("estimating transition matrix...")
    output_ = torch.tensor([]).float().cuda()

    # collect all the outputs
    with torch.no_grad():
        for batch_idx, (idx, data, label) in enumerate(tqdm(train_dataloader)):
            data = torch.autograd.Variable(data.cuda())
            outputs = model_ce(data)
            outputs = torch.tensor(outputs).float().cuda()
            outputs = F.softmax(outputs, dim=1)
            output_ = torch.cat([output_, outputs], dim=0)

    # find argmax_{x^i} p(y = e^i | x^i) for i in C
    hard_instance_index = output_.argmax(dim=0)

    trans_mat_ = torch.tensor([]).float()

    # T_ij = p(y = e^j | x^i) for i in C j in C
    for i in range(num_classes):
        trans_mat_ = torch.cat([trans_mat_, output_[hard_instance_index[i]].cpu()], dim=0)

    trans_mat_ = trans_mat_.reshape(num_classes, num_classes)

    return trans_mat_


def FW_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)

    outputs = F.softmax(output, dim=1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs)
    loss = CE_loss(outputs, target)
    return loss

def main_fw():
    model = torch.load('./model_ce_' + str(args.r) + '_' + str(args.seed))
    trans_mat = find_trans_mat(model)
    best_valid_loss = validate(valid_loader=val_dataloader, model=model, criterion=FW_loss, trans_mat=trans_mat)
    print('valid_loss=', best_valid_loss)
    torch.save(model, './model_fw_' + str(args.r) + '_' + str(args.seed))
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(args.epochs):
        print("epoch=", epoch)
        print('r=', args.r)
        print("traning model_fw...")
        learning_rate=0.0001
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        train(train_loader=train_dataloader, model=model, optimizer=optimizer, criterion=FW_loss, trans_mat=trans_mat)
        print("validating model_fw...")
        valid_loss = validate(valid_loader=val_dataloader, model=model, criterion=FW_loss, trans_mat=trans_mat)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, './model_fw_' + str(args.r) + '_' + str(args.seed))
            print("saved.")

def initialize(model):
    print('initializing infer_z, noisy_y and est_c...')

    output_ = torch.tensor([]).float().cuda()
    infer_z = torch.tensor([]).long().cuda()
    noisy_y = torch.tensor([]).long().cuda()

    with torch.no_grad():
        for batch_idx, (idx, data, label) in enumerate(tqdm(train_dataloader_unshuffle)):
            data = torch.autograd.Variable(data.cuda())
            label = torch.tensor(label).long().cuda()
            outputs = F.softmax(model(data), dim=1)
            outputs = torch.tensor(outputs).float().cuda()

            output_ = torch.cat([output_, outputs], dim=0)
            infer_z_ = outputs.max(1)[1]
            infer_z = torch.cat([infer_z, infer_z_], dim=0)
            noisy_y = torch.cat([noisy_y, label], dim=0)

    est_C = torch.zeros((num_classes, num_classes)).cuda()

    for idx in range(noisy_y.size(0)):

        est_C[:,int(noisy_y[idx])] += output_[idx]

    return infer_z, noisy_y, est_C

def Approximate_Gibbs_sampling(outputs,labels,T):
    T_ = T.transpose(1,0)
    unnorm_probs = outputs * T_[labels.long()]
    probs = unnorm_probs / (unnorm_probs.sum(1).unsqueeze(1))
    latent_label = torch.distributions.categorical.Categorical(probs=probs).sample()
    return latent_label

def train_lccn(model, infer_z, noisy_y, est_C):
    best_test_acc = test(model, test_loader=val_dataloader)
    torch.save(model, './model_lccn_' + str(args.r) + '_' + str(args.seed))
    epoch = args.epochs
    alpha = 1.0
    step  = 0
    warming_up_step = 1000
    freq_trans = 100
    trans_warmup = (est_C + alpha) / (est_C + alpha).sum(1).unsqueeze(1)
    trans = torch.zeros((num_classes,num_classes)).cuda()
    C = est_C

    for epoch_idx in range(epoch):
        print("epoch:", epoch_idx)
        print("r=", args.r)
        model.train()
        learning_rate = 0.0001
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)

        print('training model_lccn...')
        for batch_idx, (index, data, label) in enumerate(tqdm(train_dataloader)):
            if index.size(0) != batch_size:
                continue

            data = torch.autograd.Variable(data.cuda())
            label = torch.tensor(label).long().cuda()
            index = torch.tensor(index).long().cuda()

            if step % freq_trans == 0 and step != 0:  # update transition matrix in each feq_trans steps
                trans = (C + alpha) / (C + alpha).sum(1).unsqueeze(1)
            if step < warming_up_step:
                T = trans_warmup
            else:
                T = trans

            optimizer.zero_grad()
            outputs = model(data)

            # sampling latent approximate true label
            latent_labels = Approximate_Gibbs_sampling(F.softmax(outputs.detach(),1), label, T).detach().long().cuda()
            loss = CE_loss(outputs, latent_labels)
            loss.backward()

            optimizer.step()

            step += 1

            for i in range(index.size(0)):
                index_ = index[i]
                if infer_z[index_] != 0 and infer_z[index_] != 1:
                    print(infer_z[index_])
                    exit()
                C[infer_z[index_]][noisy_y[index_]] -= 1
                infer_z[index_] = latent_labels[i]
                C[infer_z[index_]][noisy_y[index_]] += 1

        print('validating model_lccn...')
        test_acc=test(model, test_loader=val_dataloader)
        print('val_acc=', test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model, './model_lccn_' + str(args.r) + '_' + str(args.seed))
            print('saved.')

def main_lccn():
    model_lccn = torch.load('./model_ce_' + str(args.r) + '_' + str(args.seed)).cuda()
    infer_z, noisy_y, est_C = initialize(model_lccn)
    train_lccn(model_lccn, infer_z, noisy_y, est_C)

def evaluate(path):
    print(args.r, args.seed, path)
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_dataloader)
    print('test_acc=', test_acc)


main_ce()
main_dmi()
main_gce()
main_fw()
main_lccn()
evaluate('./model_ce_' + str(args.r) + '_' + str(args.seed))
evaluate('./model_dmi_' + str(args.r) + '_' + str(args.seed))
evaluate('./model_gce_' + str(args.r) + '_' + str(args.seed))
evaluate('./model_fw_' + str(args.r) + '_' + str(args.seed))
evaluate('./model_lccn_' + str(args.r) + '_' + str(args.seed))
