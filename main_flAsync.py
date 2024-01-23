#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sys

from utils.gen_cnn import CNNMnist, CNNMnistSglLyr
from utils.gen_cnn import CNNCifar, CNNCifar1, CNNCifar2
from utils.data_allocation import dataAlloc, getDataset, loadDataAlloc, loadDataHist
from utils.sys_wireless import gen_channel
from utils.sys_wireless import compress, compute_q, cmpr_init
from utils.rslt import test_img, printVec, plotting, rst
from utils.eval import grt_print

from fl_algo.mdlTrainAggr import LocalUpdate, adaptiveLr, train_allAg
from fl_algo.mdlTrainAggr import mdl_aggr
from fl_algo.sch import scheduleUser, cpct_sym
from fl_algo.sys_async import getUpdateAgentSeq

import copy
import math
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # mode, scheduling and aggregation
    parser.add_argument('--case', type=int, default=0, help="0: benchmark, 1: indepTrain, 2: FedAvg, 3: Asym-FedAvg, 4:fedAsync") 
    parser.add_argument('--epochs', type=int, default=20, help="global iteration num") 
    parser.add_argument('--maxCmpTime', type=float, default=4.0, help="maximal computation time of agent")
    parser.add_argument('--chkPeriod', type=float, default=1.0, help="periodic check time in asyncFL")
    parser.add_argument('--lamda', type=float, default=0.02, help="constraint param. in local loss function")
    parser.add_argument('--schMode', type=int, default=0, help="0: random scheduling; 1: significance; 2: frequency; 3: loss-based; 4: test method")
    parser.add_argument('--switchEpochSch4', type=int, default=0, help="number of initial epochs running random scheduling and then switched to schMode=4")
    parser.add_argument('--varTrialNum', type=int, default=30, help="trial num for minimization of variance in sch 9-11")
    parser.add_argument('--denumType', type=int, default=0, help="0: R(|pi|); 1: N(total agents)") 
    parser.add_argument('--aggrMode', type=int, default=0, help="0: equal weight; 1: favorFresh; 2: favorOld")
    parser.add_argument('--gamma', type=float, default=2.3, help="gamma in age metric")
    parser.add_argument('--adjEpsilon', type=float, default=1.0, help="extra scaling on epsilon in age-channel aware scheduling")

    # resource allocation
    parser.add_argument('--rscAllocMode', type=int, default=0, help="0: max-min fairness; 1: equal-slot allocation") 
    parser.add_argument('--dnmcRscAlloc', action='store_true', help="dynamic resource symbol adjustment per UL communication run")
    parser.add_argument('--numRscSym', type=int, default=5000, help="number of average resource symbol per UL communication run, based on our method") 
    
    # federated arguments
    parser.add_argument('--all_clients', type=int, default=0, help="all clients used")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.3, help="the fraction of clients: C")
    parser.add_argument('--dnmcFrac', type=int, default=0, help="dynamic C; 0: OFF; 1: ON, incremental increasing from C to 1.8C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--alpha', type=float, default=0.8, help="alpha-filtering in fedAsync")

    # compression
    parser.add_argument('--compressMode', type=int, default=0, help="0: no compression; 1: UpdateAware; 2: proposed from 1; 3: random selection and random quantization; 4: largest selection and random quantization; 5: updateAwareNew") 
    parser.add_argument('--cmprSegNum', type=int, default=2, help="Valid when compressMode=3; number of quantization levels, should be 2^n")
       
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset: mnist, cifar")
    parser.add_argument('--note', type=str, default='-', help="leave message")
    parser.add_argument('--cuda', type=str, default='cuda:0', help="cuda device index")
    parser.add_argument('--ovl_ratio', type=float, default=1.0, help='ovl ratio')
    parser.add_argument('--num_shard', type=int, default=200, help='number of shard')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--num_rlz', type=int, default=10, help="number of realization")
    parser.add_argument('--genDataAlloc', action='store_true', help="Data allocation to agents")
    parser.add_argument('--cifarMdl', type=int, default=0, help='CNN model selection for CIFAR-10')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    
    if args.case == 4 and not args.num_rlz == 1:
        print("only support num_rlz=1 for case=4, due to different realization having different update timing")
        sys.exit()
    
    if args.genDataAlloc:
        dataAlloc(args.dataset, args.num_rlz, args.case, args.ovl_ratio, args.num_users, args.num_shard)
            
    # Load training dataset and distribute it to agents
    dataset_train, dataset_test = getDataset(args.dataset)
           
    if args.case == 0:
        num_user_adpt = 1
    else:
        num_user_adpt = args.num_users  

    mode = args.schMode

    # performance evaluation 
    rst_out = rst(args.num_rlz)
    
    if args.dnmcFrac == 0:
        m = max(int(args.frac * num_user_adpt), 1)
        wtLossBased = np.array([1/m, 1-1/m]) 
    else:
        step_r = 0.05
        fracSet = np.array([args.frac])
        while fracSet[-1:] < (args.frac*1.79):
            fracSet = np.append(fracSet, fracSet[-1:] + step_r)

    Kc = max(int(0.5 * num_user_adpt), 1)
    #snr = 10000 #10
    snr = 20
    if not args.dnmcRscAlloc:
        # numSym is determined according to maxCmpTime=4
        if args.case == 2:
            numSym = args.numRscSym * args.maxCmpTime
        else:
            numSym = int(args.numRscSym * (4/args.maxCmpTime))
    
    schLossBasedatSvr = 0

    qTable = np.array([])
    paramCount = 0
    for rlz in range(args.num_rlz):       
        if args.dataset == 'mnist':
            net_glob = CNNMnist(args.num_channels, args.num_classes).to(device)
        else:
            if args.cifarMdl == 0:
                net_glob = CNNCifar(args.num_channels, args.num_classes).to(device)
            elif args.cifarMdl == 1:
                net_glob = CNNCifar1(args.num_channels, args.num_classes).to(device)
            elif args.cifarMdl == 2:
                net_glob = CNNCifar2(args.num_channels, args.num_classes).to(device)

        net_glob.train()
        # copy weights
        w_glob = net_glob.state_dict()
        
        qTable, paramCount = cmpr_init(rlz, qTable, paramCount, w_glob, args.cmprSegNum, args.compressMode)
        
        #load data allocation of agent
        dict_users = loadDataAlloc(args.dataset, args.ovl_ratio, args.num_users, args.num_shard, rlz)

        # load data histogram
        data_hist = loadDataHist(args.dataset, args.ovl_ratio, args.num_users, args.num_shard, rlz)
        
        # weight initialization for model aggregation
        mdlAggr = mdl_aggr(dict_users)

        cnt_tgl_users = np.zeros(num_user_adpt, dtype=int)    
        rst_out.loss_test_glb = 0
        trainLossBuf = np.full(num_user_adpt, np.inf) 
        
        #for printing gradient
        printGrdt = 0
        if printGrdt == 1:
            targetLyr = ['fc2.bias','conv1.bias','fc2.weight']
            grtEval = grt_print(targetLyr, num_user_adpt)
            #print(w_glob.keys())    

        if args.case <= 2:
            w_locals = [copy.deepcopy(w_glob) for i in range(num_user_adpt)] #type=list
            loss_locals = np.zeros(num_user_adpt) 

            for it in range(args.epochs):
                # simulate channel
                h = gen_channel(num_user_adpt)

                learning_rate = adaptiveLr(args.lr, it, args.epochs)
                
                if args.dnmcRscAlloc:
                    if it < (args.epochs/2):
                        numSym = int(args.numRscSym/2)
                    else:
                        numSym = int(args.numRscSym*3/2)

                if args.dnmcFrac == 1:
                    mIdx = int(it / int(args.epochs/fracSet.size)) 
                    m = max(int(fracSet[mIdx] * num_user_adpt), 1)
                    wtLossBased = np.array([1/m, 1-1/m])

                train_all = train_allAg()
                train_all.train_sys(range(num_user_adpt), args.case, net_glob, w_glob, w_locals, args.local_ep, learning_rate, args.local_bs, 0, device, dataset_train, dataset_test, dict_users, schLossBasedatSvr, wtLossBased)
                
                if it <= args.switchEpochSch4-1 and mode >= 3:
                    schModeAdpt = 0
                else:
                    schModeAdpt = copy.deepcopy(mode)
                    
                idxs_users, nC = scheduleUser(schModeAdpt, range(num_user_adpt), m, train_all.w_locals_chk, w_glob, schLossBasedatSvr, train_all.loss_locals_chk, train_all.test_loss, trainLossBuf, rst_out.loss_test_glb, cnt_tgl_users, h, Kc, numSym, snr, args.rscAllocMode, data_hist, args.adjEpsilon, args.varTrialNum, 1) 
                 
                if not args.compressMode == 0:
                    q_select_users, nzRatio = compute_q(nC, paramCount, args.compressMode, qTable)
                 
                for selAg in idxs_users:
                    cnt_tgl_users[selAg] += 1

                for qidx, idx in enumerate(idxs_users):
                    w_locals[idx] = copy.deepcopy(train_all.w_locals_chk[idx])
                    if not args.compressMode == 0:
                        w_locals[idx] = compress(w_locals[idx], w_glob, q_select_users[qidx], args.compressMode, args.cmprSegNum)
                    loss_locals[idx] = copy.deepcopy(train_all.loss_locals_chk[idx])
                
                if printGrdt == 1:
                    grtEval.comp(w_glob, w_locals, learning_rate)

                # update global weights
                w_glob = mdlAggr.cmpAggr(args.denumType, w_locals, idxs_users)
                                
                # update trainLoss buffer
                for ag in range(num_user_adpt):
                    trainLossBuf[ag] = copy.deepcopy(train_all.loss_locals_chk[ag])


                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)

                # testing
                net_glob.eval()
                rst_out.eval(net_glob, dataset_train, dataset_test, args.bs, device)
                rst_out.write(rlz)
                
                # print loss
                if not args.case == 1:
                    loss_avg = sum(loss_locals) / len(loss_locals)
                else:
                    loss_avg = sum(loss_locals) / np.count_nonzero(loss_locals)
                print('rlz {}; epoch: {}; accuracy: {}\n'.format(rlz, it, rst_out.acc_test))

        elif args.case >= 3:
            agentCmptTime = args.maxCmpTime*np.random.rand(args.num_users)
            agentCmptTime = np.sort(agentCmptTime)
            updateTierSeq, updateIterIdx = getUpdateAgentSeq(args.maxCmpTime, agentCmptTime, args.chkPeriod, args.epochs, args.case)            
            num_epoch_for_case34 = len(updateTierSeq)

            sk_t_record = -1*np.ones(num_user_adpt, dtype=int)

            w_locals = [copy.deepcopy(w_glob) for i in range(num_user_adpt)] #type=list
            w_glob_ag = [copy.deepcopy(w_glob) for i in range(num_user_adpt)]
            num_epoch_adpt = num_epoch_for_case34
            if args.case == 4:
                numSym = int(numSym/(num_epoch_adpt/args.maxCmpTime/args.epochs))

            for it in range(num_epoch_adpt):
                updateAgSeqAdpt = updateTierSeq[it]
                if not updateAgSeqAdpt:
                    rst_out.write(rlz)
                    continue

                loss_locals = []
                train_all = train_allAg()
                learning_rate = adaptiveLr(args.lr, it, num_epoch_adpt)    
                
                #simulate channel
                h = gen_channel(len(updateAgSeqAdpt))

                if args.case == 3:    
                    if args.dnmcRscAlloc:
                        if it < (num_epoch_adpt/2):
                            numSym = int(args.numRscSym/2)
                        else:
                            numSym = int(args.numRscSym*3/2)
                    if args.dnmcFrac == 1:
                        mIdx = int(it / int(num_epoch_adpt/fracSet.size)) 
                        m = max(int(fracSet[mIdx] * num_user_adpt), 1)
                        wtLossBased = np.array([1/m, 1-1/m])

                    dataNumInK = 0
                    for idx in updateAgSeqAdpt:
                        dataNumInK += len(dict_users[idx])
                
                train_all.train_sys(updateAgSeqAdpt, args.case, net_glob, w_glob, w_glob_ag, args.local_ep, learning_rate, args.local_bs, args.lamda, device, dataset_train, dataset_test, dict_users, schLossBasedatSvr, wtLossBased)
                                
                if args.case == 3:
                    if it <= args.switchEpochSch4-1 and mode >= 3:
                        schModeAdpt = 0
                    else:
                        schModeAdpt = copy.deepcopy(mode)

                    idxs_users, nC = scheduleUser(schModeAdpt, updateAgSeqAdpt, m, train_all.w_locals_chk, w_glob_ag, schLossBasedatSvr, train_all.loss_locals_chk, train_all.test_loss, trainLossBuf, rst_out.loss_test_glb, cnt_tgl_users, h, Kc, numSym, snr, args.rscAllocMode, data_hist, args.adjEpsilon, args.varTrialNum, 0)            

                    if not args.compressMode == 0:
                        q_select_users, nzRatio = compute_q(nC, paramCount, args.compressMode, qTable)

                    for selAg in idxs_users:
                        cnt_tgl_users[updateAgSeqAdpt[selAg]] += 1
                else: # asyncFL
                    idxs_users = np.zeros(1, dtype='int')
                    chGSquare = np.multiply(h.imag, h.imag) + np.multiply(h.real, h.real)
                    Cpct, _ , _  = cpct_sym(chGSquare, snr, idxs_users, np.ones(1))
                    nC = np.multiply(Cpct, numSym)
                    q_select_users, nzRatio = compute_q(nC, paramCount, args.compressMode, qTable)

                for qidx, agReIdx in enumerate(idxs_users):
                    trueIdx = updateAgSeqAdpt[agReIdx]
                    w_locals[trueIdx] = copy.deepcopy(train_all.w_locals_chk[agReIdx])
                    if not args.compressMode == 0:
                        w_locals[trueIdx] = compress(w_locals[trueIdx], w_glob_ag[trueIdx], q_select_users[qidx], args.compressMode, args.cmprSegNum)
                    loss_locals.append(copy.deepcopy(train_all.loss_locals_chk[agReIdx]))

                if args.case == 3: 
                    if printGrdt == 1:                    
                        grtEval.comp(w_glob_ag, w_locals, learning_rate)
                        
                    # update global weights
                    w_tier_aggr = mdlAggr.cmpAggrAge(args.denumType, args.gamma, it, sk_t_record, args.aggrMode, w_locals, updateAgSeqAdpt, idxs_users) 
                    
                    for idx, ag in enumerate(updateAgSeqAdpt):
                        w_glob_ag[ag] = copy.deepcopy(w_tier_aggr)

                else: # asyncFL
                    w_glob = mdlAggr.cmpAggrAsyncFl(args.alpha, w_locals[updateAgSeqAdpt[0]], w_glob)
                    for idx, ag in enumerate(updateAgSeqAdpt):
                        w_glob_ag[ag] = copy.deepcopy(w_glob)                    
                
                for idx, ag in enumerate(updateAgSeqAdpt):
                    sk_t_record[ag] = it
                    trainLossBuf[ag] = copy.deepcopy(train_all.loss_locals_chk[idx])
                
                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)

                # testing
                net_glob.eval()
                if args.case == 3:
                    net_glob.load_state_dict(w_tier_aggr)
                else: # fedAsync
                    net_glob.load_state_dict(w_glob)
                rst_out.eval(net_glob, dataset_train, dataset_test, args.bs, device)
                rst_out.write(rlz)
                print('rlz {}; epoch: {}; accuracy: {}\n'.format(rlz, it, rst_out.acc_test))
        
        if printGrdt == 1:
            print("Print vec")
            #printVec(grtEval.g_locals[0], numIter, NumElmt, agIdx)
            printVec(grtEval.g_locals[1], 1001, 1000, args.case, 3, "gradient")
            printVec(grtEval.g_locals[12], 1001, 1000, args.case, 10, "gradient")
            printVec(grtEval.g_locals[38], 1001, 1000, args.case, 35, "gradient")
            
            printVec(grtEval.g_norm[1], 1001, 1, args.case, 1, "gradientNorm")
            printVec(grtEval.g_norm[12], 1001, 1, args.case, 12, "gradientNorm")
            printVec(grtEval.g_norm[38], 1001, 1, args.case, 38, "gradientNorm")

    plotting(args, rst_out.acc_train_perRlz, rst_out.acc_test_perRlz, rst_out.loss_train_perRlz, rst_out.loss_test_perRlz)

