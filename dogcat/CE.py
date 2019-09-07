# CE - training with cross entropy loss

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from dogcat import *

CE = nn.CrossEntropyLoss().cuda()

def train(train_loader, model, optimizer, criterion=CE):

    model.train()

    for i, (idx, input, target) in enumerate(tqdm(train_loader)):
        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader):
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


def main_ce():
    model = VGG('VGG16').cuda()
    best_acc = 0

    for epoch in range(50):
        print("epoch=", epoch)
        print('r=', args.r)
        learning_rate = 1e-4
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        print("traning model_ce...")
        train(train_loader=train_loader, model=model, optimizer=optimizer)
        print("validating model_ce...")
        valid_acc = test(model=model, test_loader=val_loader)
        print('valid_acc=', valid_acc)
        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(model, './dog_model_ce_' + str(args.r) + '_' + str(args.s))
            print("saved.")


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    main_ce()
    evaluate('./dog_model_ce_' + str(args.r) + '_' + str(args.s))
