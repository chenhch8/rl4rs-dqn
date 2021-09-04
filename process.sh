#!/bin/bash
python3 src/process.py \
    --trainset dataset/trainset.csv \
    --devset dataset/track1_testset.csv \
    --testset dataset/track2_testset.csv \
    --itemset dataset/item_info.csv \
    --outdir dataset
