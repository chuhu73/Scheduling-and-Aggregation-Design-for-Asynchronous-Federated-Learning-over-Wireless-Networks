#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy

def WtConstraint(w_lcl, w_glb, lamda):
    normBuf = 0
    for k in w_lcl.keys():
        dif = w_lcl[k] - w_glb[k]
        normBuf += torch.sum(torch.mul(dif, dif))
    normBuf *= (lamda/2)       
    return normBuf

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, lr, local_bs, local_ep, rLamda, device, dataset=None, idxs=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True)
        self.lr = lr
        self.local_ep = local_ep
        self.device = device
        self.rLamda = rLamda
        
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        initGlbModel = copy.deepcopy(net.state_dict())
        
        epoch_loss = []
        for it in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                
                #reg_loss = 0
                #for param in net.parameters():
                #    reg_loss += torch.sum(torch.mul(param, param))
                reg_loss = WtConstraint(net.state_dict(), initGlbModel, self.rLamda)
                #print("original loss: ", loss)
                #print("reg_loss: ", reg_loss)
                loss += reg_loss
                
                loss.backward()
                optimizer.step()
                #if batch_idx % 10 == 0:
                #    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #        it, batch_idx * len(images), len(self.ldr_train.dataset),
                #               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def WtAvg(w, wt):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= wt[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]*wt[i]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def WtAlphaFilter(w_old, w_new, alpha):
    for k in w_old.keys():
        w_old[k] = w_old[k] * (1 - alpha) +  alpha * w_new[k]
    return w_old

def WtOfstUpdate(w_in, w_ofst, w_base):
    for k in w_in.keys():
        w_in[k] = w_in[k] - w_ofst[k] + w_base[k]      
    return w_in

def adaptiveLr(ori_lr, itrIdx, numItr):
    gear_num = 1
    #gear_num = 2
    rge = [1/(1<<exponent) for exponent in range(1, gear_num+1)]
    #rge = [1/2, 1/8, 1/16]
    learning_rate = ori_lr
    for gear in range(gear_num):
        thr = (1-rge[-gear-1])*numItr
        if itrIdx >= thr:
            learning_rate = ori_lr*rge[-gear-1]
            break
    return learning_rate

class train_allAg():
    def __init__(self, ):
        self.w_locals_chk = []
        self.loss_locals_chk = []
        self.test_loss = []

    def train_sys(self, agIdxSet, case, net_glob, w_glob, w_glob_ag, local_ep, lr, bs, lamda, device, dataset_train, dataset_test, dict_users, schLossBasedatSvr, wtLossBased):
        for ag in agIdxSet:
            if case == 1 or case >= 3:
                net_glob.load_state_dict(w_glob_ag[ag])
            else:
                net_glob.load_state_dict(w_glob)
            local = LocalUpdate(lr, bs, local_ep, 0, device, dataset=dataset_train, idxs=dict_users[ag]) ############
            w, loss = local.train(net=copy.deepcopy(net_glob).to(device))
            self.w_locals_chk.append(copy.deepcopy(w))
            self.loss_locals_chk.append(copy.deepcopy(loss))

            if schLossBasedatSvr == 1:
                if case == 1 or case >= 3:
                    wtForCmb = [w, w_glob_ag[ag]]
                else:
                    wtForCmb = [w, w_glob]
                wLossBased = WtAvg(wtForCmb, wtLossBased)
                net_glob.load_state_dict(copy.deepcopy(wLossBased))
                _ , test_loss_ag = test_img(net_glob, dataset_test, bs, device)
                self.test_loss.append(copy.deepcopy(test_loss_ag))

class mdl_aggr():
    def __init__(self, dict_users):
        self.dict_users = dict_users
        dataNumInN = 0
        self.numUser = len(dict_users)
        dataNumInN_vec = np.zeros(self.numUser, dtype=int)
        for idx in range(self.numUser):
            tmpLen = len(dict_users[idx])
            dataNumInN += tmpLen
            dataNumInN_vec[idx] = tmpLen
        self.dataNumInN_vec = dataNumInN_vec
        self.dataNumInN = dataNumInN

    def getWtSch(self, idxs_users):
        wt = np.zeros(self.numUser)
        for qidx, idx in enumerate(idxs_users):
            idx = int(idx)
            wt[idx] = self.dataNumInN_vec[idx]
        return wt

    def cmpAggr(self, denumType, w_locals, idxs_users):
        if denumType == 0:
            wt = self.getWtSch(idxs_users)
            wt = wt / sum(wt)
            w_glob = WtAvg(w_locals, wt)
        else:
            w_glob = WtAvg(w_locals, self.dataNumInN_vec/self.dataNumInN)

        return w_glob

    def cmpAggrAge(self, denumType, gamma, it, sk_t_record, aggrMode, w_locals, updateAgSeqAdpt, idxs_users):
        trueIdxVec = np.zeros(idxs_users.size)
        for idx, ag in enumerate(idxs_users):
            trueIdxVec[idx] = updateAgSeqAdpt[ag]
        dataNumSchl = self.getWtSch(trueIdxVec)
        dataNum_age_for_aggr = np.zeros(self.numUser)
        ageMetric = np.zeros(self.numUser)
        for ag in range(self.numUser):
            pwrTerm = it-sk_t_record[ag]-1
            if aggrMode == 1:
                ageMetric[ag] = pow(gamma,pwrTerm)/math.exp(pwrTerm)
            elif aggrMode == 2:
                ageMetric[ag] = math.exp(pwrTerm)/pow(gamma,pwrTerm)
            if denumType == 0:
                dataNum_age_for_aggr[ag] = ageMetric[ag]*dataNumSchl[ag]
            else:
                dataNum_age_for_aggr[ag] = ageMetric[ag]*self.dataNumInN_vec[ag]

        if denumType == 0:
            dataNum_for_aggr = dataNumSchl
        else:
            dataNum_for_aggr = self.dataNumInN_vec

        if aggrMode == 0: ## equal weight
            targetArray = dataNum_for_aggr
        else: # age-based weight
            targetArray = dataNum_age_for_aggr

        w_tier_aggr = WtAvg(w_locals, targetArray/sum(targetArray))
        return w_tier_aggr

    def cmpAggrAsyncFl(alpha, w_local, w_glb):
        w_glob = WtAlphaFilter(w_glb, w_local, alpha)
        return w_glob
