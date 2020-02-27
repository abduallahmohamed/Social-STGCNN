# !/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag social-stgcnn-eth --use_lrschd --num_epochs 250 && echo "eth Launched." &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag social-stgcnn-hotel --use_lrschd --num_epochs 250 && echo "hotel Launched." &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag social-stgcnn-univ --use_lrschd --num_epochs 250 && echo "univ Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag social-stgcnn-zara1 --use_lrschd --num_epochs 250 && echo "zara1 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag social-stgcnn-zara2 --use_lrschd --num_epochs 250 && echo "zara2 Launched." &
P4=$!

wait $P0 $P1 $P2 $P3 $P4