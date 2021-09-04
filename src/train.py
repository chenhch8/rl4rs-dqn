#!/usr/bin/env python
# coding=utf-8
import torch
from deepq import deepq
from env import Env
from dataset import Dataset
import argparse

def main(args):
    item_feature, item_price, item_loc = torch.load('./dataset/items_info.pt')
    trainset = Dataset(args.dataset, 256)
    env = Env(item_price, 3)
    network = {
        'item_embed': item_feature,
        'out_dim': item_feature.size(0),
        'num_layers': [3, 3],
        'user_dim': 10,
        'bundle_size': 3,
        'nhead': 1,
        'seq_length': 356,
        'dropout': 0.1,
        'dueling': bool(args.dueling)
    }
    deepq.learn(env,
                network,
                trainset,
                item_loc,
                lr=args.lr,
                checkpoint_path=args.outdir,
                total_epochs=args.total_epochs,
                batch_size=args.bz,
                buffer_size=args.bs,
                print_freq=10,
                target_network_update_freq=args.target_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--dataset', default='./dataset/train.pt', type=str)
    parser.add_argument('--total_epochs', default=5000, type=int)
    parser.add_argument('--bz', default=1024, type=int)
    parser.add_argument('--target_num', default=10, type=int)
    parser.add_argument('--bs', default=100000, type=int)
    parser.add_argument('--dueling', default=1, type=int)
    args = parser.parse_args()
    main(args)
