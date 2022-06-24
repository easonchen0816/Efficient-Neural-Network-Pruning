"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch

from config import cfg
import os
import json
import numpy as np

class Logger():
    def __init__(self, cfg, overwrite=True):
        self.cfg = cfg
        self.base_path = os.path.join('./logs', cfg.base.task_name)
        self.logfile = os.path.join(self.base_path, 'log.txt')
        self.cfgfile = os.path.join(self.base_path, 'cfg.json')

        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
            if not os.path.isfile(self.logfile) or overwrite:
                with open(self.logfile, 'w') as fp:
                    fp.write('')
            with open(self.cfgfile, 'w') as fp:
                json.dump(cfg.raw(), fp)

    def save_record(self, epoch, record):
        with open(self.logfile) as fp:
            log = json.load(fp)

        log[str(epoch)] = record
        with open(self.logfile, 'w') as fp:
            json.dump(log, fp)

    def save_info(self, info, info_fn):
        infofile = os.path.join(self.base_path, info_fn)
        with open(infofile, 'wb') as f:
            pickle.dump(info, f)
    
    def save_log(self, _str):
        with open(self.logfile, 'a') as fp:
            fp.write(_str+'\n')