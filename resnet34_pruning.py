import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''import sys

_r = os.getcwd().split('/')
_p = '/'.join(_r[:_r.index('tailor_by_gbn')+1])
print('Change dir from %s to %s' % (os.getcwd(), _p))
os.chdir(_p)
sys.path.append(_p)'''

from config import parse_from_dict
parse_from_dict({
    "base": {
        "task_name": "resnet34_caltech256_ticktock",
        "cuda": True,
        "seed": 0,
        "checkpoint_path": "",
        "epoch": 0,
        "multi_gpus": True,
        "fp16": False
    },
    "model": {
        "name": "ir_resnet34",
        "num_class": 257,
        "pretrained": True
    },
    "train": {
        "trainer": "normal",
        "max_epoch": 40,
        "optim": "sgd",
        "steplr": [
            [10, 1.0],
            [30, 0.2],
            [40, 0.04]
        ],
        "weight_decay": 5e-3,
        "momentum": 0.9,
        "nesterov": False
    },
    "data": {
        "type": "caltech256", 
        "shuffle": True,
        "batch_size": 125,
        "test_batch_size": 125,
        "num_workers": 4
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "ratio_i": 0.3,
        "lr_min": 1e-4,
        "lr_max": 1e-2,
        "tock_epoch": 40,
        "T": 10,
        "p":0.002
    }
})
from config import cfg

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from logger import Logger
from models import get_model
from loader import get_loader
from loader.imagenet import get_imagenet
from trainer import get_trainer
from utils import dotdict
from loss import get_criterion

from prune.universal_ir import Meltable, GatedBatchNorm2d, Conv2dObserver, IterRecoverFramework, FinalLinearObserver
from prune.utils import analyse_model, finetune
import random
import math


"""----"""
def _sgdr(epoch):
    lr_min, lr_max = cfg.train.sgdr.lr_min, cfg.train.sgdr.lr_max
    restart_period = cfg.train.sgdr.restart_period
    _epoch = epoch - cfg.train.sgdr.warm_up

    while _epoch/restart_period > 1.:
        _epoch = _epoch - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(_epoch/restart_period)
    return lr_min + (lr_max - lr_min) *  0.5*(1.0 + math.cos(radians))

def _step_lr(epoch):
    v = 0.0
    for max_e, lr_v in cfg.train.steplr:
        v = lr_v
        if epoch <= max_e:
            break
    return v

def get_lr_func():
    if cfg.train.steplr is not None:
        return _step_lr
    elif cfg.train.sgdr is not None:
        return _sgdr
    else:
        assert False

def adjust_learning_rate(epoch, pack):
    if pack.optimizer is None:
        if cfg.train.optim == 'sgd' or cfg.train.optim is None:
            pack.optimizer = optim.SGD(
                pack.net.parameters(),
                lr=1,
                momentum=cfg.train.momentum,
                weight_decay=cfg.train.weight_decay,
                nesterov=cfg.train.nesterov
            )
        else:
            print('WRONG OPTIM SETTING!')
            assert False
        pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())

    pack.lr_scheduler.step()
    return pack.lr_scheduler.get_lr()
def set_optim(pack):
    if pack.optimizer is None:
        if cfg.train.optim == 'sgd' or cfg.train.optim is None:
            pack.optimizer = optim.SGD(
                [
                {'params': pack.net.module.layer1.parameters()},
                {'params': pack.net.module.layer2.parameters()},
                {'params': pack.net.module.layer3.parameters()},
                {'params': pack.net.module.layer4.parameters()},
                {'params': pack.net.module.fc_c.parameters(), 'lr': 0.005}
            ],
                lr=0.0005,
                momentum=cfg.train.momentum,
                weight_decay=cfg.train.weight_decay,
                nesterov=cfg.train.nesterov
            )
        else:
            print('WRONG OPTIM SETTING!')
            assert False
        pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())


def recover_pack():
    train_loader, test_loader = get_loader()
    train_loader_i, test_loader_i = get_imagenet()

    pack = dotdict({
        'net': get_model(),
        'train_loader': train_loader,
        'test_loader': test_loader,
        'trainer': get_trainer(),
        'criterion': get_criterion(),
        'optimizer': None,
        'lr_scheduler': None,
        'logger': Logger(cfg),
        'best': None,
        'train_loader_i': train_loader_i,
        'test_loader_i': test_loader_i
    })
    print(pack.net)
    set_optim(pack)
    #adjust_learning_rate(cfg.base.epoch, pack)
    return pack

def set_seeds():
    torch.manual_seed(cfg.base.seed)
    if cfg.base.cuda:
        torch.cuda.manual_seed_all(cfg.base.seed)
        torch.backends.cudnn.deterministic = True
        if cfg.base.fp16:
            torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.base.seed)
    random.seed(cfg.base.seed)
import uuid

def bottleneck_set_group(net):
    layers = [
        net.module.layer1,
        net.module.layer2,
        net.module.layer3,
        net.module.layer4
    ]
    for m in layers:
        masks = []
        if m == net.module.layer1:
            masks.append(pack.net.module.bn1)
        for mm in m.modules():
            if mm.__class__.__name__ == 'BasicBlock':
                
                if mm.downsample is not None and len(mm.downsample._modules) > 0:
                    #print(m, mm)
                    masks.append(mm.downsample._modules['1'])
                masks.append(mm.bn2)

        group_id = uuid.uuid1()
        for mk in masks:
            mk.set_groupid(group_id)
#print(pack.net.module)
set_seeds()
pack = recover_pack()

GBNs = GatedBatchNorm2d.transform(pack.net)
for gbn in GBNs:
    gbn.extract_from_bn()

bottleneck_set_group(pack.net)

def clone_model(net):
    model = get_model()
    gbns = GatedBatchNorm2d.transform(model.module)
    model.load_state_dict(net.state_dict())
    return model, gbns

cloned, _ = clone_model(pack.net)
BASE_FLOPS, BASE_PARAM = analyse_model(cloned.module, torch.randn(1, 3, 32, 32).cuda())
print('%.3f MFLOPS' % (BASE_FLOPS / 1e6))
print('%.3f M' % (BASE_PARAM / 1e6))
del cloned

def eval_prune(pack):
    cloned, _ = clone_model(pack.net)
    _ = Conv2dObserver.transform(cloned.module)
    cloned.module.fc_c = FinalLinearObserver(cloned.module.fc_c)
    cloned_pack = dotdict(pack.copy())
    cloned_pack.net = cloned
    Meltable.observe(cloned_pack, 0.001)
    Meltable.melt_all(cloned_pack.net)
    flops, params = analyse_model(cloned_pack.net.module, torch.randn(1, 3, 32, 32).cuda())
    del cloned
    del cloned_pack
    
    return flops, params

"""----"""

pack.trainer.test(pack)

pack.tick_trainset = pack.train_loader
prune_agent = IterRecoverFramework(pack, GBNs, cfg, ratio_i = cfg.gbn.ratio_i)
prune_agent.set_score_i()
flops_save_points = set([90, 80, 70, 60, 50, 40])
iter_idx = 0
flops_ms = 90
prune_agent.tock(tock_epoch=20)
#prune_agent.freeze_model()
# prune_agent.finetune_i(10)
#prune_agent.recover_model()

while True:
    left_filter = prune_agent.total_filters - prune_agent.pruned_filters
    num_to_prune = int(left_filter * cfg.gbn.p)
    info = prune_agent.prune(num_to_prune, tick=True, lr=cfg.gbn.lr_min)
    flops, params = eval_prune(pack)
    info.update({
        'flops': '[%.2f%%] %.3f MFLOPS' % (flops/BASE_FLOPS * 100, flops / 1e6),
        'param': '[%.2f%%] %.3f M' % (params/BASE_PARAM * 100, params / 1e6)
    })
    _str = ('Iter: %d,\t FLOPS: %s,\t Param: %s,\t Left: %d,\t Pruned Ratio: %.2f %%,\t Train Loss: %.4f,\t Test Acc: %.2f' % 
          (iter_idx, info['flops'], info['param'], info['left'], info['total_pruned_ratio'] * 100, info['train_loss'], info['after_prune_test_acc']))
    print(_str)
    pack.logger.save_log(_str)
    flops_ratio = flops/BASE_FLOPS * 100
    iter_idx += 1
    #if iter_idx % cfg.gbn.T == 0:
    if flops_ratio <= flops_ms:
        flops_ms = flops_ms-10
        #prune_agent.alpha_to_beta()
        print('Tocking:')
        set_optim(prune_agent.pack)
        prune_agent.tock(tock_epoch=cfg.gbn.tock_epoch)
        # break
        #prune_agent.beta_to_alpha()

    
    for point in [i for i in list(flops_save_points)]:
        if flops_ratio <= point:
            torch.save(pack.net.module.state_dict(), './logs/resnet34_caltech256_ticktock/%s.ckp' % str(point))
            flops_save_points.remove(point)

    if len(flops_save_points) == 0:
        break

"""### You can see how to fine-tune and get the pruned network in the finetune.ipynb"""

