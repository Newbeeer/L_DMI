# DMI - training with L_DMI

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from dogcat import *

CE = nn.CrossEntropyLoss().cuda()

def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


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


def validate(valid_loader, model, criterion):

    model.eval()
    loss_t = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(tqdm(valid_loader)):
            if i == 1:
                break

            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            output = model(input)
            loss = criterion(output, target)
            loss_t += loss

    print('valid_loss=', loss_t.item())
    return loss_t

def main_dmi():
    model = torch.load('./dog_model_ce_' + str(args.r) + '_' + str(args.s))
    best_valid_loss = validate(valid_loader=val_loader, model=model, criterion=DMI_loss)
    print('valid_loss=', best_valid_loss)
    torch.save(model, './dog_model_dmi_' + str(args.r) + '_' + str(args.s))

    for epoch in range(50):
        print("epoch=", epoch)
        print('r=', args.r)
        learning_rate = 1e-5
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)

        print("traning model_dmi...")
        train(train_loader=train_loader, model=model, optimizer=optimizer, criterion=DMI_loss)
        valid_loss = validate(valid_loader=val_loader, model=model, criterion=DMI_loss)
        print('valid_loss=', valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, './dog_model_dmi_' + str(args.r) + '_' + str(args.s))
            print("saved")


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader)
    print('final_test_acc=', test_acc)


if __name__ == '__main__':
    main_dmi()
    evaluate('./dog_model_dmi_' + str(args.r) + '_' + str(args.s))
