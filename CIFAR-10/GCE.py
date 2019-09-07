# GCE - training with generalized cross entropy loss

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import *
from dataset import *

num_classes = 10

CE = nn.CrossEntropyLoss().cuda()

set_pruning = False

q = 0.7
k = 0.5
w_pruning = torch.tensor([1] * train_dataset_noisy.__len__())

def lq_loss(outputs, target):
    # loss = (1 - h_j(x)^q) / q
    loss = 0

    for i in range(outputs.size(0)):
        loss += (1.0 - (outputs[i][target[i]]) ** q) / q

    loss = loss / outputs.size(0)

    return loss


def train(model, optimizer, pruning):
    if pruning == True:
        with torch.no_grad():
            for batch_idx, (idx, data, target) in enumerate(train_loader_noisy):
                data = torch.tensor(data).float().cuda()
                target = torch.tensor(target).cuda()
                outputs = model(data)
                outputs = F.softmax(outputs, dim=1)
            for i in range(outputs.size(0)):
                if outputs[i][target[i]] <= k:
                   w_pruning[idx[i]] = 0

    model.train()

    for i, (idx, input, target) in enumerate(train_loader_noisy):
        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        outputs = model(input)
        outputs = F.softmax(outputs, dim=1)

        loss = lq_loss(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



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


def test(model, test_loader=test_loader_):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (idx, input, target) in enumerate(test_loader):
            input = torch.Tensor(input).cuda()
            target = torch.autograd.Variable(target).cuda()

            total += target.size(0)
            output = model(input)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def main_gce():
    model_lq = torch.load('./model_ce_' + str(args.r) + '_' + str(args.s)).cuda()
    best_valid_loss = validate(model_lq, valid_loader_noisy, lq_loss)
    torch.save(model_lq, './model_gce_' + str(args.r) + '_' + str(args.s))

    for epoch in range(120):
        print("epoch=", epoch,'r=', args.r)
        pruning = False
        learning_rate = 1e-2
        if epoch >= 40:
            learning_rate = 1e-3
        if epoch >= 80:
            learning_rate = 1e-4
        if set_pruning == True and epoch >= 40 and epoch % 10 == 0:
            pruning = True

        optimizer_lq = torch.optim.SGD(model_lq.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        train(model_lq, optimizer_lq, pruning)

        valid_loss = validate(model_lq, valid_loader_noisy, lq_loss)
        #test_acc = test(model=model_lq, test_loader=test_loader_)
        #print('test_acc=', test_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_lq, './model_gce_' + str(args.r) + '_' + str(args.s))
            print('saved.')


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    print("GCE")
    main_gce()
    evaluate('./model_gce_' + str(args.r) + '_' + str(args.s))
