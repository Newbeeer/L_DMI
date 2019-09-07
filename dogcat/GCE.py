# GCE: training with generalized cross entropy loss

import torch.nn.parallel
import torch.optim
import torch.utils.data
from dogcat import *

num_classes = 2

CE = nn.CrossEntropyLoss().cuda()

set_pruning = False

q = 0.7

def lq_loss(outputs, target):
    # loss = (1 - h_j(x)^q) / q
    loss = 0

    for i in range(outputs.size(0)):
        loss += (1.0 - (outputs[i][target[i]]) ** q) / q

    loss = loss / batch_size

    return loss


def train(model, optimizer):
    model.train()

    for i, (idx, input, target) in enumerate(tqdm(train_loader)):
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
        for i, (idx, input, target) in enumerate(tqdm(test_loader)):
            input = torch.Tensor(input).cuda()
            target = torch.autograd.Variable(target).cuda()

            output = model(input)
            output = F.softmax(output, dim=1)
            loss = criterion(output, target)
            loss_t += loss

    return loss_t


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


def main():
    model_lq = torch.load('./dog_model_ce_' + str(args.r) + '_' + str(args.s)).cuda()
    torch.save(model_lq, './dog_model_lq_' + str(args.r) + '_' + str(args.s))

    valid_loss = validate(model_lq, val_loader, lq_loss)
    best_valid_loss = valid_loss

    for epoch in range(50):
        print('epoch:', epoch)
        learning_rate = 1e-3

        optimizer_lq = torch.optim.SGD(model_lq.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        print("traning model_lq...")
        train(model_lq, optimizer_lq)

        print("validating model_lq...")
        valid_loss = validate(model_lq, val_loader, lq_loss)
        print(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_lq, './dog_model_lq_' + str(args.r) + '_' + str(args.s))
            print('saved.')


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader)
    print('final_test_acc=', test_acc)


if __name__ == '__main__':
    main()
    evaluate('./dog_model_lq_' + str(args.r) + '_' + str(args.s))

