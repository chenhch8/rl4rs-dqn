import argparse
import torch
import numpy as np
import pandas as pd
import os
import random


def normalize(array, axis=0):
    _min = array.min(axis=axis, keepdims=True)
    _max = array.max(axis=axis, keepdims=True)
    factor = _max - _min
    return (array - _min) / np.where(factor != 0, factor, 1)


def parse_click_history(history_list):
    clicks = list(map(lambda user_click: list(map(lambda item: item.split(':')[0],
                                                  user_click.split(','))),
                      history_list))
    _max_len = max(len(items) for items in clicks)
    clicks = [items + [0] * (_max_len - len(items)) for items in clicks]
    clicks = torch.tensor(np.array(clicks, dtype=np.long)) - 1
    return clicks


def parse_user_protrait(protrait_list):
    return torch.tensor(normalize(np.array(list(map(lambda x: x.split(','),
                                                    protrait_list)),
                                           dtype=np.float32)))


def process_item(filename, outdir):
    item_info = pd.read_csv(filename, ' ')
    item2id = np.array(item_info['item_id']) - 1
    item2loc = torch.tensor(np.array(item_info['location'], dtype=np.float32)[item2id])
    item2price = torch.tensor(normalize(np.array(item_info['price'], dtype=np.float32)[item2id]) * 10, dtype=torch.float32)
    item2feature = torch.tensor(normalize(np.array(list(map(lambda x: x.split(','),
                                               item_info['item_vec'])),
                                      dtype=np.float32)[item2id]))
    item2info = torch.cat([item2feature, item2price[:, None], item2loc[:, None]], dim=-1)
    torch.save([item2info, item2price, item2loc], os.path.join(outdir, 'items_info.pt'))


def process_data(filename, outdir, savename):
    dataset = pd.read_csv(filename, ' ')
    click_items = parse_click_history(dataset['user_click_history'])
    user_protrait = parse_user_protrait(dataset['user_protrait'])
    exposed_items = None
    if 'exposed_items' in dataset.columns:
        exposed_items = torch.tensor(np.array(list(map(lambda x: x.split(','),
                                                       dataset['exposed_items'])),
                                              dtype=np.long) - 1)
    torch.save([user_protrait, click_items, exposed_items],
               os.path.join(outdir, savename))


def main(args):
    print('processing items info ...')
    process_item(args.itemset, args.outdir)
    print('processing trainset ...')
    process_data(args.trainset, args.outdir, 'train.pt')
    print('processing devset ...')
    process_data(args.devset, args.outdir, 'dev.pt')
    print('processing testset ...')
    process_data(args.testset, args.outdir, 'test.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', required=True, type=str)
    parser.add_argument('--devset', required=True, type=str)
    parser.add_argument('--testset', required=True, type=str)
    parser.add_argument('--itemset', required=True, type=str)
    parser.add_argument('--outdir', required=True, type=str)
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedir(args.outdir, exists_ok=True)

    main(args)
