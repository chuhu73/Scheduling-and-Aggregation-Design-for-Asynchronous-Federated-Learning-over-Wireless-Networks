#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sys
import copy

sys.path.append('../fl_algo')
from fl_algo.sch import WtSignificant 

class grt_print:
    def __init__(self, targetLyr, num_user):
        self.targetLyr = targetLyr
        self.g_locals= [[] for i in range(num_user)]
        self.g_norm = [[] for i in range(num_user)]
        self.num_user = num_user

    def comp(self, w_glob, w_locals, learning_rate):
        for ag in range(self.num_user):
            for i in range(len(self.targetLyr)):
                grdt = torch.sub(w_glob[targetLyr[i]], w_locals[ag][targetLyr[i]])
                grdt = torch.div(grdt, learning_rate)
                if i==0:
                    nparray = copy.deepcopy(grdt.detach().to('cpu').numpy())
                    nparray = nparray.reshape(-1)
                else:
                    arr = copy.deepcopy(grdt.detach().to('cpu').numpy())
                    nparray = np.concatenate((nparray, arr.reshape(-1)))
            self.g_locals[ag].append(nparray)
                
        for ag in range(self.num_user):
            normInUpdate = WtSignificant(w_locals, w_glob, range(self.num_user), 1)
            normInUpdate = normInUpdate/(learning_rate ** 2)
            self.g_norm[ag].append(normInUpdate)

