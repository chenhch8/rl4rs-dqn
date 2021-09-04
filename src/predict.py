#!/usr/bin/env python
# coding=utf-8
import torch
from deepq import deepq
from env import Env
from dataset import Dataset
from tqdm import tqdm
import json

import pdb

def main(filename, outfile):
    item_feature, item_price, item_loc = torch.load('./dataset/items_info.pt')
    dataset = Dataset(f'./dataset/{filename}')
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
        'dueling': False
    }
    act = deepq.learn(env,
                      network,
                      trainset=dataset,
                      act_mask=item_loc,
                      checkpoint_path='./output/',
                      total_epochs=0)

    result = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    act_mask = item_loc.to(device)
    for user, click_items, click_mask, exposed_items in tqdm(dataset):
        obs = (user, click_items, click_mask, None, None)
        for t in range(3):
            act_, _ = act(obs,
                          eps_greedy=0,
                          is_greedy=True)
            new_obs = env.new_obs(obs, act_)
            pre_obs, obs = obs, new_obs
        #pdb.set_trace()
        result.extend(obs[3].view(-1, 9).tolist())
    with open(f'./output/{outfile}', 'w') as fw:
        json.dump(result, fw)

if __name__ == '__main__':
    main('test.pt', 'test_result.json')
    main('dev.pt', 'dev_result.json')
