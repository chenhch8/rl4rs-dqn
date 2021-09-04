#!/usr/bin/env python
# coding=utf-8
import torch
import random

import pdb

class Dataset:
    def __init__(self, filename, batch_size=1024):
        self.user_protrait, self.click_items, self.exposed_items \
                = torch.load(filename)
        self.click_mask = self.click_items != -1
        self.click_items[self.click_items == -1] = 0

        self.all_indexs = list(range(len(self.user_protrait)))
        self.cur_index = 0
        self.bz = batch_size

    def reset(self):
        random.shuffle(self.all_indexs)
        self.cur_index = 0

    def __len__(self):
        return len(self.all_indexs) // self.bz + int(bool(len(self.all_indexs) % self.bz))

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index >= len(self.all_indexs):
            raise StopIteration
    
        i = self.all_indexs[self.cur_index:self.cur_index + self.bz]
        user, click_items, click_mask = \
                self.user_protrait[i], self.click_items[i], self.click_mask[i]
        exposed_items = self.exposed_items[i] if self.exposed_items is not None else None

        self.cur_index += self.bz

        return user, click_items, click_mask, exposed_items if exposed_items is not None else None


if __name__ == '__main__':
    dataset = Dataset('./dataset/train.pt')
    print(len(dataset))
    for user_protrait, click_items, click_mask, exposed_items in dataset:
        pass
    dataset.reset()
    print('done')
    for user_protrait, click_items, click_mask, exposed_items in dataset:
        pass
    print('done')
