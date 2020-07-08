# CE - training with cross entropy loss

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import *
from dataset import *

num_classes = 10

CE = nn.CrossEntropyLoss().cuda()

def train(train_loader, model, optimizer, criterion=CE):

    model.train()

    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader=test_loader_):
    model.eval()
    correct = 0
    total = 0

    for i, (idx, input, target) in enumerate(tqdm(test_loader)):
        input = torch.Tensor(input).cuda()
        target = torch.autograd.Variable(target).cuda()

        total += target.size(0)
        output = model(input)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def main_ce():
    model_ce = ResNet34().cuda()
    best_ce_acc = 0

    for epoch in range(120):

        learning_rate = 0.1
        if epoch >= 40:
            learning_rate = 0.01
        if epoch >= 80:
            learning_rate = 0.001
        optimizer_ce = torch.optim.SGD(model_ce.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        print("traning model_ce...")
        train(train_loader=train_loader_noisy, model=model_ce, optimizer=optimizer_ce)
        print("validating model_ce...")
        valid_acc = test(model=model_ce, test_loader=valid_loader_noisy)
        print('epoch : {}, r: {}, valid_acc : {}'.format(epoch,args.r,valid_acc))
        if valid_acc >= best_ce_acc:
            best_ce_acc = valid_acc
            torch.save(model_ce, './model_ce_' + str(args.r) + '_' + str(args.s))
            print("saved.")


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('Final test_acc:', test_acc)


if __name__ == '__main__':
    main_ce()
    print("CE traning finished")
    evaluate('./model_ce_' + str(args.r) + '_' + str(args.s))
