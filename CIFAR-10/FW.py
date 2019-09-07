# FW - training with forward loss

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import *
from dataset import *

num_classes = 10

CE = nn.CrossEntropyLoss().cuda()

def find_trans_mat(model_ce):
    # estimate each component of matrix T based on training with noisy labels
    print("estimating transition matrix...")
    output_ = torch.tensor([]).float().cuda()

    # collect all the outputs
    with torch.no_grad():
        for batch_idx, (idx, data, label) in enumerate(train_loader_noisy):
            data = torch.tensor(data).float().cuda()
            outputs = model_ce(data)
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


def forward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs)
    loss = CE(outputs, target)
    return loss


def train(train_loader, model, optimizer, trans_mat):

    model.train()

    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        loss = forward_loss(output, target, trans_mat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(valid_loader, model, trans_mat):

    model.eval()
    loss_t = 0

    with torch.no_grad():

        for i, (idx, input, target) in enumerate(valid_loader):

            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            output = model(input)
            loss = forward_loss(output, target, trans_mat)
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

def main_fw():
    model_forward = torch.load('./model_ce_' + str(args.r) + '_' + str(args.s)).cuda()
    trans_mat = find_trans_mat(model_forward)
    best_loss = validate(valid_loader=valid_loader_noisy, model=model_forward, trans_mat=trans_mat)
    torch.save(model_forward, './model_fw_' + str(args.r) + '_' + str(args.s))

    for epoch in range(120):
        print("epoch=", epoch,'r=', args.r)

        learning_rate = 0.01
        if epoch >= 40:
            learning_rate = 0.001
        if epoch >= 80:
            learning_rate = 0.0001

        optimizer_forward = torch.optim.SGD(model_forward.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)

        print("traning model_forward...")
        train(train_loader=train_loader_noisy, model=model_forward, optimizer=optimizer_forward, trans_mat=trans_mat)

        print("validating model_forward...")
        valid_loss = validate(valid_loader=valid_loader_noisy, model=model_forward, trans_mat=trans_mat)
        #test_acc = test(model=model_forward, test_loader=test_loader_)
        #print('test_acc=', test_acc)
        print('valid_loss=', valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model_forward, './model_fw_' + str(args.r) + '_' + str(args.s))
            print('saved.')


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    print("FW:")
    main_fw()
    evaluate('./model_fw_' + str(args.r) + '_' + str(args.s))
