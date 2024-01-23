#!/bin/bash
# Training data need to be allocated to the users (ex. line 5~6) before running the FL process (ex. line 9 to 18)

# distribute training data to each user with i.i.d./non-i.i.d. setting
python3 ./main_flAsync.py --case 2 --genDataAlloc --num_users 40 --ovl_ratio 1 --num_rlz 10
python3 ./main_flAsync.py --case 2 --genDataAlloc --num_users 40 --ovl_ratio -1 --num_rlz 10

# non-i.i.d., FedAvg and the proposed method, random scheduling and the proposed scheduling
python3 ./main_flAsync.py --cuda 'cuda:0' --epochs 35 --frac 0.2 --num_rlz 10 --case 2 --num_users 40 --ovl_ratio -1 --local_bs 50 --schMode 0 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &
python3 ./main_flAsync.py --cuda 'cuda:1' --epochs 35 --frac 0.2 --num_rlz 10 --case 2 --num_users 40 --ovl_ratio -1 --local_bs 50 --schMode 11 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &
python3 ./main_flAsync.py --cuda 'cuda:0' --epochs 35 --frac 0.2 --num_rlz 10 --case 3 --num_users 40 --ovl_ratio -1 --local_bs 50 --schMode 0 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &
python3 ./main_flAsync.py --cuda 'cuda:1' --epochs 35 --frac 0.2 --num_rlz 10 --case 3 --num_users 40 --ovl_ratio -1 --local_bs 50 --schMode 11 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &

# i.i.d., FedAvg and the proposed method, random scheduling and the proposed scheduling
python3 ./main_flAsync.py --cuda 'cuda:1' --epochs 35 --frac 0.2 --num_rlz 10 --case 2 --num_users 40 --ovl_ratio 1 --local_bs 50 --schMode 0 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &
python3 ./main_flAsync.py --cuda 'cuda:0' --epochs 35 --frac 0.2 --num_rlz 10 --case 2 --num_users 40 --ovl_ratio 1 --local_bs 50 --schMode 11 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &
python3 ./main_flAsync.py --cuda 'cuda:0' --epochs 35 --frac 0.2 --num_rlz 10 --case 3 --num_users 40 --ovl_ratio 1 --local_bs 50 --schMode 0 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &
python3 ./main_flAsync.py --cuda 'cuda:1' --epochs 35 --frac 0.2 --num_rlz 10 --case 3 --num_users 40 --ovl_ratio 1 --local_bs 50 --schMode 11 --compressMode 3 --cmprSegNum 4 --numRscSym 300000 &

