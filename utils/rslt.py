#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

def test_img(net_g, datatest, bs, device):
    net_g.eval()
    test_loss = 0
    correct = 0
    # testing
    with torch.no_grad():
        data_loader = DataLoader(datatest, batch_size=bs)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if device.type != "cpu":
                data, target = data.to(device), target.to(device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def printVec(vec, numIter, NumElmt, case, agIdx, name):
    print(len(vec))
    print(vec[0].shape)
    rowNum = min(len(vec), numIter)
    colNum = min(vec[0].shape[0], NumElmt)
    ptArr = np.empty([rowNum, colNum])
    if (2*colNum) < vec[0].shape[0]:
        step = math.floor(vec[0].shape[0]/colNum)
    else:
        step = 1
    for ri in range(rowNum):
        ptArr[ri] = vec[ri][0:colNum*step:step]
    filename = name + "_case" + str(case) + "_agent" + str(agIdx) + ".txt"
    np.savetxt(filename, ptArr, fmt="%2.5f", delimiter=',', newline='\n', header='agent'+str(agIdx))

class rst:
    def __init__(self, num_rlz):
        self.loss_train_perRlz = [[] for i in range(num_rlz)]
        self.loss_test_perRlz = [[] for i in range(num_rlz)]
        self.acc_train_perRlz = [[] for i in range(num_rlz)]
        self.acc_test_perRlz = [[] for i in range(num_rlz)]
        self.acc_train = 0
        self.loss_train = 0
        self.acc_test = 0
        self.loss_test_glb = 0

    def eval(self, net_glob, dataset_train, dataset_test, bs, device):
        self.acc_train, self.loss_train = test_img(net_glob, dataset_train, bs, device)
        self.acc_test, self.loss_test_glb = test_img(net_glob, dataset_test, bs, device)

    def write(self, rlz):
        self.acc_train_perRlz[rlz].append(self.acc_train)
        self.acc_test_perRlz[rlz].append(self.acc_test)
        self.loss_train_perRlz[rlz].append(self.loss_train)
        self.loss_test_perRlz[rlz].append(self.loss_test_glb)

def plotting(args, acc_train_perRlz, acc_test_perRlz, loss_train_perRlz, loss_test_perRlz):
    if args.case == 2:
        acc_test_avg = np.zeros(args.epochs)
        for rlz in range(args.num_rlz):
            acc_test_avg = np.add(acc_test_avg, acc_test_perRlz[rlz])
        acc_test_avg /= args.num_rlz 

        acc_train_loss_avg = np.zeros(args.epochs)
        for rlz in range(args.num_rlz):
            acc_train_loss_avg = np.add(acc_train_loss_avg, loss_train_perRlz[rlz])
        acc_train_loss_avg /= args.num_rlz 
        acc_test_loss_avg = np.zeros(args.epochs)
        for rlz in range(args.num_rlz):
            acc_test_loss_avg = np.add(acc_test_loss_avg, loss_test_perRlz[rlz])
        acc_test_loss_avg /= args.num_rlz 

        if args.dataset == 'mnist':
            ftest2=open('avgTestEpochFedAvg.txt','a')
        else:
            ftest2=open('cifar_avgTestEpochFedAvg.txt','a')

        str1 = "msg:{} ovl: {} numUsr: {} numShd: {} frac: {} lr: {} sch: {} varNum: {} aggrMode: {} scaleEpsilon: {} nitialEpchSch4: {} cpsMd: {} numRscSym: {} ".format(args.note, args.ovl_ratio, args.num_users, args.num_shard, args.frac, args.lr, args.schMode, args.varTrialNum, args.aggrMode, args.adjEpsilon, args.switchEpochSch4, args.compressMode, args.numRscSym)
        if args.dataset == 'cifar':
            strFinal = str1 + "cifarMdl: " + str(args.cifarMdl) + "\n"
        else:
            strFinal = str1 + "\n"
        ftest2.write(strFinal)
        for epch in range(args.epochs):
            ftest2.write("{:.2f} ".format(acc_test_avg[epch]))
        ftest2.write("\n")
        ftest2.close()   
        
        if args.dataset == 'mnist':
            ftest2=open('avgTrainLossFedAvg.txt','a')
        else:
            ftest2=open('cifar_avgTrainLossFedAvg.txt','a')

        ftest2.write(strFinal)
        for epch in range(args.epochs):
            ftest2.write("{:.4f} ".format(acc_train_loss_avg[epch]))
        ftest2.write("\n")
        ftest2.close()   
        
        if args.dataset == 'mnist':
            ftest2=open('avgTestLossFedAvg.txt','a')
        else:
            ftest2=open('cifar_avgTestLossFedAvg.txt','a')

        ftest2.write(strFinal)
        for epch in range(args.epochs):
            ftest2.write("{:.4f} ".format(acc_test_loss_avg[epch]))
        ftest2.write("\n")
        ftest2.close()   
        
    elif args.case >= 3:
        acc_test_avg = np.zeros(len(acc_test_perRlz[0]))
        for rlz in range(args.num_rlz):
            acc_test_avg = np.add(acc_test_avg, acc_test_perRlz[rlz])
        acc_test_avg /= args.num_rlz      

        acc_train_loss_avg = np.zeros(len(loss_train_perRlz[0]))
        for rlz in range(args.num_rlz):
            acc_train_loss_avg = np.add(acc_train_loss_avg, loss_train_perRlz[rlz])
        acc_train_loss_avg /= args.num_rlz 
        acc_test_loss_avg = np.zeros(len(loss_test_perRlz[0]))
        for rlz in range(args.num_rlz):
            acc_test_loss_avg = np.add(acc_test_loss_avg, loss_test_perRlz[rlz])
        acc_test_loss_avg /= args.num_rlz 

        if args.dataset == 'mnist':
            ftest2=open('avgTestEpoch.txt','a')
        else:
            ftest2=open('cifar_avgTestEpoch.txt','a')
        
        str1 = "msg:{} ovl: {} case: {} numUsr: {} numShd: {} frac: {} lr: {} denumType: {} maxCmpTime: {} aggrMode: {} alpha: {} gamma: {} schMode: {} varNum: {} scaleEpsilon: {} lamda: {} cprMd: {} numRscSym: {} ".format(args.note, args.ovl_ratio, args.case, args.num_users, args.num_shard, args.frac, args.lr, args.denumType, args.maxCmpTime, args.aggrMode, args.alpha, args.gamma, args.schMode, args.varTrialNum, args.adjEpsilon, args.lamda, args.compressMode, args.numRscSym)

        if args.dataset == 'cifar':
            strFinal = str1 + "cifarMdl: " + str(args.cifarMdl) + "\n"
        else:
            strFinal = str1 + "\n"
        ftest2.write(strFinal)
        
        for epch in range(len(acc_test_perRlz[rlz])):
            ftest2.write("{:.2f} ".format(acc_test_avg[epch]))
        ftest2.write("\n")
        ftest2.close() 

        if args.dataset == 'mnist':
            ftest2=open('avgTrainLoss.txt','a')
        else:
            ftest2=open('cifar_avgTrainLoss.txt','a')
        
        ftest2.write(strFinal)
        
        for epch in range(len(loss_train_perRlz[0])):
            ftest2.write("{:.4f} ".format(acc_train_loss_avg[epch]))
        ftest2.write("\n")
        ftest2.close() 
        
        if args.dataset == 'mnist':
            ftest2=open('avgTestLoss.txt','a')
        else:
            ftest2=open('cifar_avgTestLoss.txt','a')
        
        ftest2.write(strFinal)
        
        for epch in range(len(loss_test_perRlz[0])):
            ftest2.write("{:.4f} ".format(acc_test_loss_avg[epch]))
        ftest2.write("\n")
        ftest2.close() 
           
    elif args.case <= 1:
        fLoss=open('lossLog.txt','a')
        fLoss.write("case: {} ovl: {} frac: {} lr: {}\n".format(args.case, args.ovl_ratio, args.frac, args.lr))
        fGamma=open('gammaLog.txt','a')
        fGamma.write("case: {} ovl: {} frac: {} lr: {}\n".format(args.case, args.ovl_ratio, args.frac, args.lr))
        for rlz in range(args.num_rlz):
            for epch in range(len(loss_train_perRlz[rlz])):
                fLoss.write("{:.4f} ".format(float(loss_train_perRlz[rlz][epch])))
            fLoss.write(";\n")    
        fLoss.close()
        for rlz in range(args.num_rlz):
            fGamma.write("{:.4f}\n".format(float(loss_train_perRlz[rlz][-1])))
        fGamma.close()
        # plot loss curve
        plt.figure()
        axis_x = np.linspace(0, args.epochs, num=args.epochs)
        for rlz in range(args.num_rlz):
            plt.plot(axis_x, loss_train_perRlz[rlz])
        plt.ylabel('weighted average of local loss')
        if args.case == 1:
            plt.title('Independent training')
        else:
            plt.title('Centralized training')
    
    

