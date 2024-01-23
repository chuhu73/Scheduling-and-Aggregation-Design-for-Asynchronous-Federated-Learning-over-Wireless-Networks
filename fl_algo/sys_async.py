#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math

def getUpdateAgentSeq(maxCmptTime, agentCmptTime, chkPeriod, epochs, case):    
    totalT = epochs*maxCmptTime
    numAgent = agentCmptTime.size
    if case == 3:
        numGlbItrt = math.ceil(totalT/chkPeriod)
        updateTierSeq = [[] for i in range(numGlbItrt)]
        for ag in range(numAgent):
            if agentCmptTime[ag] <= chkPeriod:
                for glbItr in range(numGlbItrt):
                    updateTierSeq[glbItr].append(ag)
            else:
                skipIterNum = math.ceil(agentCmptTime[ag]/chkPeriod)
                for glbItr in range(skipIterNum-1, numGlbItrt, skipIterNum):
                    updateTierSeq[glbItr].append(ag)
    else: # fedAsync case=4
        numGlbItrt = np.sum(np.floor(np.divide(totalT, agentCmptTime)), dtype='int')
        cand = np.zeros((numGlbItrt, 2))
        cnt = 0
        for ag in range(numAgent):
            times = 1
            while times * agentCmptTime[ag] <= totalT:
                cand[cnt, :] = [ag, times * agentCmptTime[ag]]
                times += 1
                cnt += 1
        cand = cand[cand[:,1].argsort()]

        updateTierSeq = [[] for i in range(numGlbItrt)]
        for glbItr in range(numGlbItrt):
            updateTierSeq[glbItr].append(cand[glbItr,0].astype('int'))
            
    updateIterIdx = [[-1] for i in range(numAgent)] #record global iteration index where each tier has involved in
    iterCnt = 0
    for seqIdx in updateTierSeq:
        for i in seqIdx:
            updateIterIdx[i].append(iterCnt)
        iterCnt += 1               

    return updateTierSeq, updateIterIdx

