#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import copy
import math

def gen_channel(num_user_adpt):
    hr = np.random.normal(0, math.sqrt(1/2), num_user_adpt)
    hi = np.random.normal(0, math.sqrt(1/2), num_user_adpt)
    hif = np.multiply(hi, 1j)
    h = hr + hif
    return h 

def combination(n, r):
    output = (math.factorial(n) // math.factorial(n-r)) // math.factorial(r)
    
    return output 

def cmpr_init(rlz, qTable, paramCount, w_glob, cmprSegNum, compressMode):
    if rlz == 0:
        for wi in w_glob.keys():
            paramCount += w_glob[wi].numel()

        lvlNum = cmprSegNum
        qTable = np.zeros(paramCount)
        if compressMode > 0:
            first_one = 1
            for qi in range(1, paramCount+1):
                if first_one == 1:
                    qTable[qi-1] = math.ceil(quota_by_q(paramCount, qi, compressMode, lvlNum, qTable[0]))
                    first_one = 0
                else:
                    qTable[qi-1] = math.ceil(quota_by_q(paramCount, qi, compressMode, lvlNum, qTable[qi-2]))
    return qTable, paramCount

def quota_by_q(d, q, mode, lvlNum, quota_last):
    if mode == 1:
        if q == 1:
            metric = math.log2(combination(d, q)) 
        else:
            metric = quota_last + math.log2((d-q+1) / q)
    elif mode == 2:
        if q == 1:
            metric = math.log2(combination(d, q)) + q
        else:
            c_last = quota_last - q + 1
            metric = c_last + math.log2((d-q+1) / q)  + q
    elif mode == 3 or mode == 4:
        if q == 1:
            metric = math.log2(combination(d, q)) + q + q*math.log2(lvlNum)
        else:
            c_last = quota_last - (q-1)*(1 + math.log2(lvlNum))
            metric = c_last + math.log2((d-q+1) / q) + q + q*math.log2(lvlNum)
    elif mode == 5:
        if q == 1:
            metric = math.log2(combination(d, q)) + 33*q
        else:
            c_last = quota_last - 33*(q-1)
            metric = c_last + math.log2((d-q+1) / q) + 33*q
    return metric

def compute_q(nC, paramCount, mode, qTable):
    if mode == 1: # keep one-side
       constOffset = (32 + 1)
    elif mode == 2: # keep both sides
       constOffset = (2*32)
    elif mode == 3 or mode == 4:
       constOffset = 32 # save norm
    elif mode == 5: # newly proposed by update-aware paper
       constOffset = 0 

    bitsAvailable = nC - constOffset
    numUser = nC.size
    q = np.zeros(numUser, dtype=int)
    for ag in range(numUser):
        if bitsAvailable[ag] > 0:
            for qi in range(qTable.size-1, 0, -1):
                if bitsAvailable[ag] >= qTable[qi]:
                    q[ag] = qi + 1
                    break

    nzRatio = q / paramCount
    return q, nzRatio

def assignUpdate(qElmt, keepIdx, keepVal, numElmt, shapeLs, w_out, tsrName):
    if keepVal.size == 1:
        valVec = np.full(qElmt, keepVal)
    else:
        valVec = keepVal

    for el in range(qElmt):
        fltIdx = keepIdx[el]
        tsrIdx = 0
        accuElmt = 0
        while fltIdx >= (accuElmt + numElmt[tsrIdx]):
            accuElmt += numElmt[tsrIdx]
            tsrIdx += 1
        
        shp = shapeLs[tsrIdx]
        loc = np.zeros(len(shp), dtype=int)
        dimSizeBook = np.ones(len(shp), dtype=int)
        for bi in range(len(shp)):
            for othDim in range(bi+1, len(shp), 1):
                dimSizeBook[bi] *= shp[othDim]
        IdxAtCurrentTsr = fltIdx - accuElmt
        for dn in range(len(shp)):
            tgtRefine = IdxAtCurrentTsr
            for d_before in range(0,dn,1):
                tgtRefine -= loc[d_before]*dimSizeBook[d_before] 
            loc[dn] = int(tgtRefine/dimSizeBook[dn])
        w_out[tsrName[tsrIdx]][tuple(loc)] = w_out[tsrName[tsrIdx]][tuple(loc)] + valVec[el] 
    return w_out

def compress(w, w_last, qElmt, mode, lvlNum):
    flatArray = np.array([])
    numElmt = np.zeros(len(w.keys()), dtype=int)
    w_out = copy.deepcopy(w_last)
    if qElmt == 0:
        return w_out
    gradient = copy.deepcopy(w)
    shapeLs =[]
    tsrName = []
    for ky in w.keys():
        gradient[ky] = torch.sub(w[ky], w_last[ky])
    
    for idx, ky in enumerate(w.keys()):
        npArr = gradient[ky].detach().to('cpu').numpy()
        shapeLs.append(npArr.shape)
        flat = npArr.reshape(-1)
        flatArray = np.append(flatArray, flat)
        numElmt[idx] = flat.size
        tsrName.append(ky)

    if mode == 1 or mode == 2:
        if mode == 1:
            qKeepP = qElmt
            qKeepN = qElmt
        else:
            numP = np.sum(flatArray >= 0)
            qKeepP = int(numP/flatArray.size*qElmt)
            qKeepN = qElmt - qKeepP
 
        srtIdx = np.argsort(flatArray)
        minAvg = np.mean(flatArray[srtIdx[:qKeepN]])
        maxAvg = np.mean(flatArray[srtIdx[-qKeepP:]])
    
        if mode == 1:
            if maxAvg >= np.absolute(minAvg):
                keepIdx = srtIdx[-qKeepP:]
                keepVal = maxAvg
            else:
                keepIdx = srtIdx[:qKeepN]
                keepVal = minAvg    
            w_out = assignUpdate(qElmt, keepIdx, keepVal, numElmt, shapeLs, w_out, tsrName)

        else:
            w_out = assignUpdate(qKeepP, srtIdx[-qKeepP:], maxAvg, numElmt, shapeLs, w_out, tsrName)
            w_out = assignUpdate(qKeepN, srtIdx[:qKeepN], minAvg, numElmt, shapeLs, w_out, tsrName)
            
    elif mode == 3 or mode == 4:
        if mode == 3:
            chosenIdx = np.random.choice(range(flatArray.size), qElmt, replace=False)
        else:
            numP = np.sum(flatArray >= 0)
            qKeepP = int(numP/flatArray.size*qElmt)
            qKeepN = qElmt - qKeepP
            srtIdx = np.argsort(flatArray)
            chosenIdx = np.append(srtIdx[:qKeepN], srtIdx[-qKeepP:])

        target = flatArray[chosenIdx]
        normVal = np.linalg.norm(target)
        qSegLen = normVal / lvlNum 
        qOut = np.zeros(qElmt)
        for el in range(qElmt):
            rdmNoise = np.random.rand(1) # uniformly from (0,1)
            qOut[el] = np.floor(np.absolute(target[el])/qSegLen + rdmNoise) * qSegLen * np.sign(target[el])
        w_out = assignUpdate(qElmt, chosenIdx, qOut, numElmt, shapeLs, w_out, tsrName)    
    elif mode == 5: # new in update-aware paper
        chosenIdx = np.random.choice(range(flatArray.size), qElmt, replace=False) 
        w_out = assignUpdate(qElmt, chosenIdx, flatArray[chosenIdx], numElmt, shapeLs, w_out, tsrName)    
    
    return w_out


