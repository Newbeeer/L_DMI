# DMI - training with L_DMI

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import *
from dataset import *

num_classes = 10

def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def train(train_loader, model, optimizer, criterion):

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

    for i, (idx, input, target) in enumerate(test_loader):
        input = torch.Tensor(input).cuda()
        target = torch.autograd.Variable(target).cuda()

        total += target.size(0)
        output = model(input)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def validate(valid_loader, model, criterion):

    model.eval()
    loss_t = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(valid_loader):
            if i == 1:
                break

            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            output = model(input)
            loss = criterion(output, target)
            loss_t += loss

    #print('valid_loss=', loss_t.item())
    return loss_t

def validate_acc(valid_loader, model, criterion):

    model.eval()
    loss_t = 0
    correct = 0.0
    total = 0.0
    with torch.no_grad():

        for i, (idx, input, target) in enumerate(valid_loader):
            if i == 1:
                break
            total += target.size(0)
            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            output = model(input)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total

    return accuracy


def main_dmi():
    model_dmi = torch.load('./model_ce_' + str(args.r) + '_' + str(args.s))
    best_valid_loss = validate_acc(valid_loader=valid_loader_noisy, model=model_dmi, criterion=DMI_loss)
    torch.save(model_dmi, './model_dmi_' + str(args.r) + '_' + str(args.s))
    test_acc = test(model=model_dmi, test_loader=test_loader_)
    print('test_acc=', test_acc)
    for epoch in range(100):
        learning_rate = 1e-6

        optimizer_dmi = torch.optim.SGD(model_dmi.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)

        train(train_loader=train_loader_noisy, model=model_dmi, optimizer=optimizer_dmi, criterion=DMI_loss)

        valid_loss = validate_acc(valid_loader=valid_loader_noisy, model=model_dmi, criterion=DMI_loss)
        test_acc = test(model=model_dmi, test_loader=test_loader_)
        print('epoch : {}, r: {}, test_acc : {}'.format(epoch,args.r,test_acc))
        if valid_loss > best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_dmi, './model_dmi_' + str(args.r) + '_' + str(args.s))
            print("saved.")


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    print("DMI:")
    main_dmi()
    evaluate('./model_dmi_' + str(args.r) + '_' + str(args.s))
