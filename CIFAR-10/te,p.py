# import os
# import time
# import argparse
# import shutil
# import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import shutil
import math

# from tensorboardX import SummaryWriter

from lib.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from lib.data import get_dataset
from lib.net_measure import measure_model

from models.mobilenet_v2 import MobileNetV2_prescreen, eps, Mask, mb2_prune_ratio

from torch.autograd import Variable

from tqdm import tqdm
import time
from collections import OrderedDict
import copy


# import setGPU

def parse_args():
    parser = argparse.ArgumentParser(description='Preprune for mbv2')

    # model and data
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')

    # seed
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')

    # intermediate finetune schedule
    parser.add_argument('--lr', default=2.5e-3, type=float, help='learning rate for intermediate finetune')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')  # default 128
    parser.add_argument('--lr_type', default='fixed', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=0.1, type=float, help='number of epochs for intermediate finetune')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument('--n_gpu', default=8, type=int, help='number of GPUs to use')
    parser.add_argument('--n_worker', default=8, type=int, help='number of data loader worker')

    # adding neuron
    parser.add_argument('--num_evaluate', default=50, type=float,
                        help='num of neuron to evaluate for every evaluation. (Randomly pickup num_evaluate number of neuron if there are more potential neuron that can be add)')

    # load and save
    parser.add_argument('--load_path', default='./checkpoint', type=str,
                        help='pretrain model path to prune')
    parser.add_argument('--save_path', default='./checkpoint', type=str, help='path the save the prunde model')

    # skip for convergence criterion
    parser.add_argument('--top1_tol', default=0.02, type=float, help='tol for loss')
    parser.add_argument('--skip_eval_converge', default=0.05, type=float,
                        help='when bacth_top1 < (1 - skip_eval_convergence) * init_top, we skip eval the convergence')
    parser.add_argument('--skip', default=200, type=int, help='#skip when eval trainset for convergence criterion')
    parser.add_argument('--isfullnetpruned', default=0, type=int, help='whether to use pruned net as fullnet')

    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')

    return parser.parse_args()


torch.set_printoptions(precision=10)
criterion = nn.CrossEntropyLoss()


def get_model(path, n_class):
    from models.mobilenet_v2 import MobileNetV2
    fullnet = MobileNetV2(num_classes=1000)

    # fullnet not pruned
    if not args.isfullnetpruned:
        fullnet.load_state_dict(torch.load(args.load_path))
        net = MobileNetV2_prescreen(fullnet)
        del fullnet
    # fullnet is pruned model: used for iterative pruning
    else:
        if args.isfullnetpruned:
            net = MobileNetV2_prescreen(fullnet)
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k[0:6] == 'module':
                    name = k[7:]  # remove module.
                else:
                    name = k
                new_state_dict[name] = v

            net.load_state_dict(new_state_dict)
            del fullnet

    if args.n_gpu > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
        net = net.to(device)
    else:
        net = net.to(device)

    return net


def decide_candidate_set(m, prunable_neuron, num_evaluate=50):
    # only randomly pickup num_evaluate number of neurons to form the candidate set
    candidate_plus = []

    tem_a = m.prune_a.data.squeeze().cpu().numpy()
    tem_a = np.where(tem_a == 0)[0]  # randomly pick up outside neuron to add
    np.random.shuffle(tem_a)
    tem_a = set(tem_a)
    prunable_neuron = set(np.where(prunable_neuron.astype(float) > 0)[0])
    tem_a = list(tem_a & prunable_neuron)

    candidate_plus = tem_a[:num_evaluate_large]

    return candidate_plus


def decide_candidate(datas, targets, m, candidate_plus):
    # decide the candidate to perform update by 1/n stepsize

    datas = datas.to(device)
    opt_index = -1
    opt_loss = float('inf')
    opt_stepsize = 0.

    current_num_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))

    for candidate in candidate_plus:
        m.init_lsearch(candidate)
        m.prune_lsearch.data += 1. / (current_num_neuron + 1)

        with torch.no_grad():
            outputs = net(datas)
            loss = criterion(outputs, targets)

        if loss < opt_loss:
            opt_index = candidate
            opt_loss = loss
            opt_stepsize = 1. / (current_num_neuron + 1)

    m.update_alpha(opt_index, opt_stepsize)


def prune_a_layer(m):
    isalladd = 0
    num_layer = m.layer_num

    init_loss, init_top1, init_top5 = eval_train(net, eval_train_loader)
    print('Layer: ({:d}); Init Loss: {:.4f}; Init top1: ({:.4f}%); Init top5: ({:.4f}%)'.format(
        num_layer, init_loss, init_top1, init_top5))

    m.switch_mode('prune')

    # prunable neuron list; only consider the neuron that is inside at initial
    prunable_neuron = (m.prune_a.cpu().data.squeeze().numpy() > 0)
    all_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))

    m.empty_all_eps()

    is_first_neuron = 1
    iteration = 0
    verbose = True
    while 1:
        # get a mini-batch of data
        for datas, data_labels in train_loader: break

        with torch.no_grad():
            datas = datas.to(device)
            data_labels = data_labels.to(device)
            targets = data_labels

        candidate_plus = decide_candidate_set(m, prunable_neuron, num_evaluate=args.num_evaluate)
        decide_candidate(datas, targets, m, candidate_plus)
        outputs = net(datas)
        batch_top1, batch_top5 = accuracy(outputs.data, data_labels.data, topk=(1, 5))

        if batch_top1 >= (1. - args.skip_eval_converge) * init_top1:
            # evaluate whether converged
            cur_loss, cur_top1, cur_top5 = eval_train(net, eval_train_loader)
            cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            if verbose:
                print('Converge Eval------', args.top1_tol)
                print(
                    'Layer: ({:d}); Cur Loss: {:.4f}; Init Loss: {:.4f}; Cur top1: ({:.4f}%); Init top1: {:.4f}'.format(
                        num_layer, cur_loss, init_loss, cur_top1, init_top1))
                print('Cur_neuron/ All neuron', cur_neuron, m.scale)

            if cur_top1 >= (1. - args.top1_tol) * (init_top1): break  # reach convergence

        else:
            cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            if verbose:
                print('Layer: ({:d}); Batch top1: {:.4f}'.format(num_layer, batch_top1))
                print('Cur_neuron/ All neuron', cur_neuron, all_neuron)

        if cur_neuron >= all_neuron:
            print('all the neurons are added')
            m.set_alpha_to_init(prunable_neuron)
            isalladd = 1
            break

    print("This layer's Neuron", cur_neuron)
    cur_loss, cur_top1, cur_top5 = eval_train(net, eval_train_loader)
    print('Layer (before finetune): ({:d}); Cur Loss: {:.4f}; Cur top1: ({:.4f}%); Cur top5: ({:.4f}%)'.format(
        num_layer, cur_loss, cur_top1, cur_top5))
    print('=' * 90)

    a_para = m.prune_a.data
    a_num = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
    cur_loss, cur_top1, cur_top5 = eval_train(net, eval_train_loader)
    m.set_alpha_to_init(prunable_neuron)

    return a_para, a_num, cur_top1, isalladd


def net_prune(layer_):
    net.eval()

    # add important tuning parameter to save
    argu_dict = {
        'load_path': args.load_path,
        'top1_tol': args.top1_tol,
        'skip_eval_converge': args.skip_eval_converge,
        'layer_num': 0,
        'cfg': [],
        'ori_cfg': [],
        'pruned': [],
    }

    # get full cfg
    full_cfg = []
    cur_cfg = []
    for block_idx, block in enumerate(net.module.features if args.n_gpu > 1 else net.features):
        for m in block.mask_list:
            full_cfg.append(m.prune_a.shape[1])
            cur_cfg.append(m.prune_a.shape[1])

    no_prune_list = [0, 1, 33, 34]

    num_layer = -1
    total_start = time.time()
    for block_idx, block in enumerate(net.module.features if args.n_gpu > 1 else net.features):
        mask_count = -1
        for m in block.mask_list:

            num_layer += 1
            isalladd = 0

            # skip layers in no_prune_list
            if num_layer in no_prune_list:
                continue

            a_para, a_num, global_cur_top1, isalladd = prune_a_layer(m)
            m.prune_a.data = a_para
            cur_neuron = a_num

            print('=' * 38, ' All Finish ', '=' * 38)
            print("This layer's Neuron", cur_neuron)
            fullflops, pruneflops, fullparams, pruneparams = mb2_prune_ratio(net)
            print("Full Flops, Prune Flops, Full Params, Prune Params")
            print(fullflops, pruneflops, fullparams, pruneparams)

            cur_cfg[num_layer] = cur_neuron
            cur_loss, cur_top1, cur_top5 = eval_train(net, eval_train_loader)

            # layer finetune
            m.switch_mode('train')
            if not isalladd:
                train(train_loader, args.n_epoch)

            cur_loss, cur_top1, cur_top5 = eval_train(net, eval_train_loader)
            print('Layer (After finetune): ({:d}); Cur Loss: {:.4f}; Cur top1: ({:.4f}%); Cur top5: ({:.4f}%)'.format(
                num_layer, cur_loss, cur_top1, cur_top5))
            print('=' * 90)
            all_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))

            argu_dict['layer_num'] = num_layer
            delta = all_neuron - cur_neuron
            argu_dict['cfg'].append(cur_neuron)
            argu_dict['pruned'].append(delta)
            argu_dict['ori_cfg'] = all_neuron

            print("current cfg", argu_dict['cfg'])
            print('neuron pruned', argu_dict['pruned'])

            torch.save({'state_dict': net.state_dict(), 'argu_dict': argu_dict, },
                       os.path.join(args.save_path,
                                    'mbv2_prune_{}_{}.pth.tar'.format(args.top1_tol, args.isfullnetpruned)))

    print('total time', time.time() - total_start)
    print('Finish Prune')
    m.switch_mode('train')
    argu_dict['layer_num'] = num_layer
    torch.save({'state_dict': net.state_dict(), 'argu_dict': argu_dict, },
               os.path.join(args.save_path, 'mbv2_prune_{}_{}.pth.tar'.format(args.top1_tol, args.isfullnetpruned)))


def train(train_loader, n_epoch):
    net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    count_threshold = n_epoch * len(train_loader)
    count = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        # progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
        #             .format(losses.avg, top1.avg, top5.avg))
        if batch_idx % 200 == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        count += 1
        if count >= count_threshold:
            break
    net.eval()


def test(epoch, test_loader, save=False):
    global best_accd
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    device0 = 'cuda'
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                # inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = inputs.to(device0), targets.to(device0)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    return top1.avg


def eval_train(net, train_loader):
    net.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    count_threshold = float('inf')
    count = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)

            loss = global_imit_loss(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_tmp')
    memory_gpu = [int(x.split()[2]) for x in open('gpu_tmp', 'r').readlines()]
    memory_gpu = np.array(memory_gpu)
    print(memory_gpu)
    gpu_list = list(memory_gpu.argsort()[-args.n_gpu:][::-1])
    print(gpu_list)

    gpu_list = [int(idx) for idx in gpu_list]
    gpu_list_ = ",".join(str(i) for i in gpu_list)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_

    device = torch.device('cuda', int(gpu_list[0]))  # where to put pruning net

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    train_loader, eval_train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
                                                                       data_root=args.data_root, skip=args.skip)

    net = get_model(args.load_path, n_class)  # real training

    # criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start pruning...')
        print('Pruning {} on {}...'.format(args.model, args.dataset))
        net_prune(layer)