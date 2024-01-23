#!/usr/bin/env python
# coding: utf-8

from torchvision import datasets, transforms
import numpy as np
import sys
import os

def loadDataAlloc(dataset, ovl_ratio, num_users, num_shard, rlz):
    f_name = "./data/" + dataset + "_allocOvl" + str(ovl_ratio) + "_numUsr" + str(num_users) + "_numShd" + str(num_shard) + "_rlz" + str(rlz) + ".dat"
    reader = open(f_name,"r")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    f1 = reader.readlines()
    user_cnt = 0
    for li in f1:
        listSplit = li.split(',')
        del listSplit[-1]
        tgtArray = np.array(listSplit)
        dict_users[user_cnt] = tgtArray.astype(np.int64)
        user_cnt += 1
    reader.close()
    return dict_users

def loadDataHist(dataset, ovl_ratio, num_users, num_shard, rlz):
    f_name = "./data/" + dataset + "_allocOvl" + str(ovl_ratio) + "_numUsr" + str(num_users) + "_hist_numShd" + str(num_shard) + "_rlz" + str(rlz) + ".dat"
    reader = open(f_name,"r")
    data_hist = {i: np.array([], dtype='int64') for i in range(num_users)}
    f1 = reader.readlines()
    user_cnt = 0
    for li in f1:
        listSplit = li.split(',')
        del listSplit[-1]
        tgtArray = np.array(listSplit)
        data_hist[user_cnt] = tgtArray.astype(np.int64)
        user_cnt += 1
    reader.close()
    return data_hist

def getDataset(dataset):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    else: #cifar
        dataset_train = datasets.CIFAR10('data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('data/cifar/', train=False, download=True, transform=trans_cifar)
    return dataset_train, dataset_test

def dataAlloc(dataset, num_rlz, case, ovl_ratio, num_users, num_shard):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
    else: #cifar
        dataset_train = datasets.CIFAR10('data/cifar/', train=True, download=True, transform=trans_cifar)

    # sample users
    for rlz in range(num_rlz):
        if case == 0:
            dict_users = mnist_iid(dataset_train, 1)
        elif ovl_ratio >= 0:
            dict_users, data_hist = mnist_ovl(dataset_train, num_users, ovl_ratio)
        else:
            dict_users, data_hist = mnist_noniid(dataset_train, num_users, num_shard)

        f_name = "./data/" + dataset + "_allocOvl" + str(ovl_ratio) + "_numUsr" + str(num_users) + "_numShd" + str(num_shard) + "_rlz" + str(rlz) + ".dat"
        if os.path.isfile(f_name):
            os.remove(f_name)
        writer = open(f_name, "a+")
        for ag in range(len(dict_users)):
            np.savetxt(writer, dict_users[ag], fmt="%05d", newline=',')
            writer.write('\n')
        writer.close()
        f_name_hist = "./data/" + dataset + "_allocOvl" + str(ovl_ratio) + "_numUsr" + str(num_users) + "_hist_numShd" + str(num_shard) + "_rlz" + str(rlz) + ".dat"
        if os.path.isfile(f_name_hist):
            os.remove(f_name_hist)
        writer = open(f_name_hist, "a+")
        for ag in range(len(dict_users)):
            np.savetxt(writer, data_hist[ag], fmt="%05d", newline=',')
            writer.write('\n')
        writer.close()

    sys.exit()


# i.i.d. distribution
def mnist_iid(dataset, num_users): 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
    return dict_users

# non-i.i.d. distribution
def mnist_noniid(dataset, num_users, num_shards): 
    num_imgs = int(len(dataset) / num_shards)
    numShdPerUser = int(num_shards/num_users)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #labels = dataset.targets.numpy()   
    labels = np.array(dataset.targets)
    data_hist = [np.zeros(10) for i in range(num_users)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, numShdPerUser, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        for j in dict_users[i]:
            data_hist[i][labels[j]] = data_hist[i][labels[j]] + 1
    
    return dict_users, data_hist

# non-i.i.d. distribution with overlapping ratio
def mnist_ovl(dataset, num_users, ovl_ratio):
    num_class = 10
    num_users_per_grp = int(num_users/2)
    num_shard_per_grp = 5 + int(num_class*ovl_ratio/2.0)
    num_ovl_grp = (num_shard_per_grp-5)*2
    labels = np.array(dataset.targets)
    tagCnt = [0] * num_class
    for lbl in labels:
        tagCnt[lbl] += 1
    idxs = np.arange(len(dataset))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    data_hist = [np.zeros(10) for i in range(num_users)]
    num_shard_nonOvl = 10-num_shard_per_grp
    num_img_1stGrp, num_img_2ndGrp, num_range_1stGrp, num_range_2ndGrp = 0, 0, 0, 0
    for j in range(num_shard_nonOvl):
        num_img_1stGrp += tagCnt[j]
        num_img_2ndGrp += tagCnt[num_class-1-j]
    for k in range(num_shard_per_grp - num_shard_nonOvl):
        num_img_1stGrp += int(tagCnt[num_shard_nonOvl+k]/2)    
        num_img_2ndGrp += int(tagCnt[num_class-1-k-num_shard_nonOvl]/2)
    for l in range(num_shard_per_grp):
        num_range_1stGrp += tagCnt[l]
        num_range_2ndGrp += tagCnt[num_class-1-l]
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    num_img_per_user_1stGrp = int(num_img_1stGrp/num_users_per_grp)
    idxsSel = np.arange(len(dataset))
    for i in range(num_users_per_grp):
        rand_set = set(np.random.choice(idxsSel[0:num_range_1stGrp-1-num_img_per_user_1stGrp*i], num_img_per_user_1stGrp, replace=False))
        dict_users[i] = np.concatenate((dict_users[i], idxs[np.array(list(rand_set))]), axis=0)
        idxsSel = list(set(idxsSel) - rand_set)
    numRmInIdxsSel = 0
    for elmt in idxsSel:
        if elmt < len(dataset) - num_range_2ndGrp:
            numRmInIdxsSel += 1
    if numRmInIdxsSel > 0:
        idxsSel = list(set(idxsSel) - set(idxsSel[0:numRmInIdxsSel-1]))

    num_img_per_user_2ndGrp = int(len(idxsSel)/num_users_per_grp)
    for i in range(num_users_per_grp):
        rand_set = set(np.random.choice(idxsSel, num_img_per_user_2ndGrp, replace=False))
        dict_users[num_users_per_grp+i] = np.concatenate((dict_users[num_users_per_grp+i], idxs[np.array(list(rand_set))]), axis=0)        
        idxsSel = list(set(idxsSel) - rand_set)
    
    for i in range(num_users):
        for j in dict_users[i]:
            data_hist[i][labels[j]] = data_hist[i][labels[j]] + 1
    
    return dict_users, data_hist

