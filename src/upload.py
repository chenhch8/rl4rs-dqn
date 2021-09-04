#!/usr/bin/env python
# coding=utf-8
import json
import pandas
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', required=True, type=str)
parser.add_argument('--true_file', default='./dataset/track2_testset.csv')
parser.add_argument('--out_file', type=str, default='submission.csv')
args = parser.parse_args()

track2_testset = pd.read_csv(args.true_file, ' ')
pred_data = json.load(open(args.pred_file, 'r'))
print(len(track2_testset['user_id']), len(pred_data))
assert len(track2_testset['user_id']) == len(pred_data)
with open(args.out_file, 'w') as fr:
    for idx, data in zip(track2_testset['user_id'], pred_data):
        fr.write(str(idx) + ',' + ' '.join(list(map(lambda x: str(x + 1), data))) + '\n')
print(f'saving in {args.out_file}')
