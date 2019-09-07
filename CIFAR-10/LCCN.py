# LCCN - training with Latent Class-Conditional Noise model

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import *
from dataset import *

num_classes = 10

CE = nn.CrossEntropyLoss().cuda()
loss_lccn = CE
def Approximate_Gibbs_sampling(outputs,labels,T):
    T_ = T.transpose(1,0)
    unnorm_probs = outputs * T_[labels.long()]
    probs = unnorm_probs / (unnorm_probs.sum(1).unsqueeze(1))
    latent_label = torch.distributions.categorical.Categorical(probs=probs).sample()
    return latent_label


def test(model, test_loader=test_loader_):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(tqdm(test_loader)):
            input = torch.Tensor(input).cuda()
            target = torch.autograd.Variable(target).cuda()

            total += target.size(0)
            output = model(input)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total

    return accuracy

def validate(model, test_loader, criterion):
    model.eval()
    loss_t = 0

    with torch.no_grad():
        for i, (idx, input, target) in enumerate(test_loader):
            input = torch.Tensor(input).cuda()
            target = torch.autograd.Variable(target).cuda()

            output = model(input)
            output = F.softmax(output, dim=1)
            loss = criterion(output, target)
            loss_t += loss

    return loss_t


def initialize(model):
    print('initializing infer_z, noisy_y and est_c...')

    output_ = torch.tensor([]).float().cuda()
    infer_z = torch.tensor([]).long().cuda()
    noisy_y = torch.tensor([]).long().cuda()

    with torch.no_grad():
        for batch_idx, (idx, data, label) in enumerate(tqdm(train_loader_noisy_unshuffle)):
            data = torch.tensor(data).float().cuda()
            label = torch.tensor(label).long().cuda()
            outputs = F.softmax(model(data), dim=1)

            output_ = torch.cat([output_, outputs], dim=0)
            infer_z_ = outputs.max(1)[1]
            infer_z = torch.cat([infer_z, infer_z_], dim=0)
            noisy_y = torch.cat([noisy_y, label], dim=0)

    est_C = torch.zeros((num_classes, num_classes)).cuda()

    for idx in range(noisy_y.size(0)):

        est_C[:,int(noisy_y[idx])] += output_[idx]

    return infer_z, noisy_y, est_C


def train_lccn(model, infer_z, noisy_y, est_C):
    best_test_acc = 0
    epoch = 120
    alpha = 1.0
    step  = 0
    warming_up_step = 20000
    freq_trans = 200
    trans_warmup = (est_C + alpha) / (est_C + alpha).sum(1).unsqueeze(1)
    trans = torch.zeros((num_classes,num_classes)).cuda()
    C = est_C
    best_valid_loss = validate(model, valid_loader_noisy,criterion=CE)
    for epoch_idx in range(epoch):
        print("epoch=", epoch_idx,'r=', args.r)
        model.train()
        learning_rate = 0.5
        if epoch_idx >= 40:
            learning_rate = 0.1
        elif epoch_idx >= 80:
            learning_rate = 0.01

        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)

        for batch_idx, (index, data, label) in enumerate(train_loader_noisy):
            if index.size(0) != batch_size:
                continue

            data = torch.tensor(data).float().cuda()
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
            loss = CE(outputs, latent_labels)
            loss.backward()

            optimizer.step()

            step += 1

            for i in range(index.size(0)):
                index_ = index[i]
                C[infer_z[index_]][noisy_y[index_]] -= 1
                infer_z[index_] = latent_labels[i]
                C[infer_z[index_]][noisy_y[index_]] += 1



        valid_loss = validate(model, valid_loader_noisy,criterion=CE)
        #test_acc = test(model=model_lccn, test_loader=test_loader_)
        #print('test_acc=', test_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, './model_lccn_' + str(args.r) + '_' + str(args.s))
            print('saved.')
            test_acc = test(model=model, test_loader=test_loader_)
            print('test_acc=', test_acc)


def main_lccn():
    model_lccn = torch.load('./model_ce_' + str(args.r) + '_' + str(args.s)).cuda()
    acc = test(model=model_lccn)
    print('test_acc=', acc)
    infer_z, noisy_y, est_C = initialize(model_lccn)
    train_lccn(model_lccn, infer_z, noisy_y, est_C)


if __name__ == '__main__':
    print("LCCN:")
    main_lccn()
    model_lccn = torch.load('model_lccn_' + str(args.r) + '_' + str(args.s)).cuda()
    acc = test(model=model_lccn)
    print('test_acc=', acc)
