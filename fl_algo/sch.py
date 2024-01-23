#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sys
sys.path.append('../utils')
from utils.sys_wireless import combination  

def WtSignificant(w_lcl, w_glb, agSet, uniGlb):
    K = len(agSet)
    sgnfcNorm = np.zeros(K)
    for i in range(K):
        for k in w_lcl[0].keys():
            if uniGlb == 1:
                dif = w_lcl[i][k] - w_glb[k]
            else:
                dif = w_lcl[i][k] - w_glb[agSet[i]][k]
            sgnfcNorm[i] += torch.sum(torch.mul(dif, dif))  
            
    return sgnfcNorm

def cpct_sym(chGainSquare, snr, select_user, sgnfcNorm):
    Cpct = np.log2(1 + snr*chGainSquare[select_user])
    numSelect = len(select_user)

    cn2_mul_all = 1.0
    c_mul_all = 1.0
    for ag in range(numSelect):
        cn2_mul_all *= (Cpct[ag] * sgnfcNorm[select_user[ag]])
        c_mul_all *= Cpct[ag]

    cn2_all_except_self = np.zeros(numSelect)
    c_all_except_self = np.zeros(numSelect)
    for ag in range(numSelect):
        val1 = Cpct[ag]*sgnfcNorm[select_user[ag]]
        val2 = Cpct[ag]
        if val1 == 0 or val2 == 0:
            continue
        else:
            cn2_all_except_self[ag] = cn2_mul_all / val1
            c_all_except_self[ag] = c_mul_all / val2
    
    return Cpct, c_all_except_self, cn2_all_except_self

def sltAllocation(allocMode, numSym, C_all_except_self):
    if allocMode == 0: # max-min fairness
        dnmnt = np.sum(C_all_except_self)
        nsym = C_all_except_self/dnmnt*numSym             
    else: #equal-slot allocation
        numSelect = C_all_except_self.size 
        nsymVal = int(numSym/numSelect)
        nsym = np.full(numSelect, nsymVal)
        rcdlNumSym = numSym - nsymVal * numSelect
        if rcdlNumSym > 0:
            luckyUserSet = np.random.choice(range(numSelect), rcdlNumSym, replace=False)
            nsym[luckyUserSet] = nsym[luckyUserSet] + 1 
    return nsym


def scheduleUser(mode, agSet, R, w_lcl, w_glb, flagLossBasedAtSvr, train_loss, test_loss, train_loss_glb, test_loss_glb, cnt_tgl_users, h, Kc, numSym, snr, rscAllocMode, data_hist, adjEpsilon, varTrialNum, uniGlb):    
    K = len(agSet)
    numSelect = min(K, R)
    protect_Kc = max(numSelect, Kc)
    protect_Kc = min(K, protect_Kc)
    cntVec = np.zeros(K)
    for ag in range(K):
        cntVec[ag] = cnt_tgl_users[agSet[ag]]
    chGainSquare = np.multiply(h.imag, h.imag) + np.multiply(h.real, h.real)
    
    if mode == 0 or (mode == 2 and sum(cntVec) == 0):  # random scheduling
        select_users = np.random.choice(range(K), numSelect, replace=False)
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)       
        nC = np.multiply(Cpct, nsym)

    elif mode == 1:  # significance selection
        sgnfcNorm = WtSignificant(w_lcl, w_glb, agSet, uniGlb)
        select_users = np.argsort(sgnfcNorm)        
        select_users = select_users[-numSelect:]
        
        Cpct, C_all_except_self, cn2_all_except_self = cpct_sym(chGainSquare, snr, select_users, sgnfcNorm)

        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)       
        nC = np.multiply(nsym, Cpct)
        
    elif mode == 2: # frequency-based scheduling
        select_users = np.argsort(cntVec)
        select_users = select_users[:numSelect]
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)
        nC = np.multiply(Cpct, nsym)

    elif mode == 3 or mode == 4 or mode == 7 or mode == 8: # loss-based scheduling
        select_LB = min(round(R/4), K) 
 
        if mode == 3 or mode == 4:
            if flagLossBasedAtSvr == 1: # loss-based scheduling based on testing data
                delta = np.array(test_loss) - test_loss_glb
            else: # loss-based scheduling based on training data
                delta = np.zeros(K)
                for ag in range(K):
                    delta[ag] = train_loss[ag] - train_loss_glb[agSet[ag]]           
        else:
            sort_users = np.argsort(chGainSquare)
            sort_users = sort_users[-protect_Kc:]
            delta = np.zeros(protect_Kc)
            if flagLossBasedAtSvr == 1: # loss-based scheduling based on testing data
                for idx, ag in enumerate(sort_users):
                    delta[idx] = test_loss[ag] - test_loss_glb[ag] 
            else: # loss-based scheduling based on training data
                for idx, ag in enumerate(sort_users):
                    delta[idx] = train_loss[ag] - train_loss_glb[agSet[ag]]

        deltaNeg = np.where(delta > 0, 0, delta)
        numNeg = np.count_nonzero(deltaNeg)

        if mode == 3 or mode == 4:
            select_users = np.argsort(delta)
        else:
            select_users = sort_users[np.argsort(delta)]

        if mode == 4 or mode == 8:
            select_users = select_users[:numSelect]
        else:
            if numNeg >= numSelect:
                select_users = select_users[:numSelect]
            elif numNeg < numSelect and numNeg >= select_LB:
                select_users = select_users[:numNeg]
            else:
                select_users = select_users[:select_LB]
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)
         
        nC = np.multiply(Cpct, nsym)

    elif mode == 5: # best-channel (BC)
        select_users = np.argsort(chGainSquare)        
        select_users = select_users[-numSelect:]

        # bit quota
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)   
        nC = np.multiply(Cpct, nsym)
 
    elif mode == 6: # best-channel best-norm (BC-BN2)
        select_users = np.argsort(chGainSquare)
        select_users = select_users[-protect_Kc:]
        chosenAgSet = []
        for ui in select_users:
            chosenAgSet.append(agSet[ui])
        sgnfcNorm = WtSignificant(w_lcl, w_glb, chosenAgSet, uniGlb)
        refine_users = np.argsort(sgnfcNorm)        
        select_users = select_users[refine_users[-numSelect:]]
        sgnfcNorm = WtSignificant(w_lcl, w_glb, agSet, uniGlb)
        
        Cpct, C_all_except_self, cn2_all_except_self = cpct_sym(chGainSquare, snr, select_users, sgnfcNorm)

        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)       
        nC = np.multiply(nsym, Cpct)

    elif mode == 9 or mode == 10: # data-aware scheduling / joint-data and channel scheduling
        Cpct = np.log2(1 + snr*chGainSquare)
        epsilon = pow(6000/len(data_hist), 2)/(np.sum(Cpct)*numSelect/K)*adjEpsilon
        if mode == 9:
            withCh = 0
        else:
            withCh = 1
        select_users = dataVarPlusCh(agSet, numSelect, data_hist, Cpct, withCh, epsilon, varTrialNum)
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)       
        nC = np.multiply(Cpct, nsym)
    
    elif mode == 11:
        select_users = np.argsort(chGainSquare)
        select_users = select_users[-protect_Kc:]
        chosenAgSet = []
        for ui in select_users:
            chosenAgSet.append(agSet[ui])
        bestSet = dataVarPlusCh(chosenAgSet, numSelect, data_hist, np.zeros(K), 0, 0, varTrialNum)
        select_users = select_users[bestSet]
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)       
        nC = np.multiply(Cpct, nsym)
    elif mode == 12: #age-based reference
        select_users = np.argsort(chGainSquare)
        select_users = select_users[-protect_Kc:]
        chosenAgSet = []
        for ui in select_users:
            chosenAgSet.append(agSet[ui])
        cntVec1 = np.zeros(protect_Kc)
        for ag in range(protect_Kc):
            cntVec1[ag] = cnt_tgl_users[chosenAgSet[ag]]

        rfIdx = np.argsort(cntVec1)
        select_users = select_users[rfIdx[:numSelect]]
        Cpct, C_all_except_self, _  = cpct_sym(chGainSquare, snr, select_users, np.ones(K))
        #print("Cpct: ", Cpct)
        nsym = sltAllocation(rscAllocMode, numSym, C_all_except_self)
        nC = np.multiply(Cpct, nsym)

    #print("select_users: ", select_users)    

    return select_users, nC

def dataVarPlusCh(agSet, numSelect, data_hist, cpct, withCh, epsilon, varTrialNum):
    K = len(agSet)
    varChLen = min(varTrialNum, combination(K, numSelect))
    varCh = np.zeros(varChLen)
    cmbSet = np.zeros((varChLen, numSelect), dtype='i')
    for ci in range(varChLen):
        cmbSet[ci,:] = np.random.choice(range(K), numSelect, replace=False)

    maxCpct = 0
    maxVar = 0

    for ci in range(varChLen):
        b_bar = 0
        sumBk = np.zeros(10)
        for ag in cmbSet[ci,:]:
           b_bar = b_bar + np.sum(data_hist[agSet[ag]])
           sumBk = np.add(sumBk, data_hist[agSet[ag]])
        b_bar = b_bar/10
        norm_val = np.linalg.norm(np.subtract(sumBk, np.full(10, b_bar)))
        if withCh == 0:
            varCh[ci] = pow(norm_val, 2)/pow(numSelect, 2)
        else:
            cpctSum = 0
            for ag in cmbSet[ci,:]:
                cpctSum = cpctSum + cpct[ag]
            if cpctSum > maxCpct:
                maxCpct = cpctSum
            if pow(norm_val, 2)/pow(numSelect, 2) > maxVar:
                maxVar = pow(norm_val, 2)/pow(numSelect, 2)
            varCh[ci] = pow(norm_val, 2)/pow(numSelect, 2) - epsilon*cpctSum
    optCmbIdx = np.argmin(varCh)
    select_users = cmbSet[optCmbIdx,:]

    return select_users


