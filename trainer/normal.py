"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from config import cfg

FINISH_SIGNAL = 'finish'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NormalTrainer():
    def __init__(self):
        self.use_cuda = cfg.base.cuda

    def test(self, pack, topk=(1,)):
        pack.net.eval()
        loss_acc, correct, total = 0.0, 0.0, 0.0
        hub = [[] for i in range(len(topk))]

        for data, target in pack.test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = pack.net(data)
                loss_acc += pack.criterion(output, target).data.item()
                acc = accuracy(output, target, topk)
                for acc_idx, score in enumerate(acc):
                    hub[acc_idx].append(score[0].item())

        loss_acc /= len(pack.test_loader)
        info = {
            'test_loss': loss_acc
        }
        
        for acc_idx, k in enumerate(topk):
            info['acc@%d' % k] = np.mean(hub[acc_idx])
        #print('Test Loss: %.4f,\t Test Acc: %.2f' % (info['test_loss'], info['acc@1']))
        return info

    def train(self, pack, update=True, iter_hook=None, mute=False, acc_step=1, max_iter=192):
        pack.net.train()
        loss_acc, correct_acc, total = 0.0, 0.0, 0.0
        begin = time()

        pack.optimizer.zero_grad()
        with tqdm(total=max_iter, disable=mute) as pbar:
            #print(len(pack.train_loader), max_iter)
            for cur_iter, (data, label) in enumerate(pack.train_loader):
                '''if cur_iter == 0:
                    print('dataset:', pack.net.module.mod, 'data:', data[0], 'label:',label)'''
                if iter_hook is not None:
                    signal = iter_hook(cur_iter, max_iter)
                    if signal == FINISH_SIGNAL:
                        break
                
                if self.use_cuda:
                    data, label = data.cuda(), label.cuda()
                data = Variable(data, requires_grad=False)
                label = Variable(label)
                logits = pack.net(data)
                loss = pack.criterion(logits, label)
                loss = loss / acc_step
                loss.backward()
                if update:
                    pack.optimizer.step()
                pack.optimizer.zero_grad()

                loss_acc += loss.item()
                pbar.update(1)
                if cur_iter == max_iter-1:
                    break

        info = {
            'train_loss': loss_acc / max_iter,
            'epoch_time': time() - begin
        }
        return info
