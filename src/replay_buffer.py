#!/usr/bin/python3
# coding: utf-8
import random
import math
import numpy as np
import torch
from collections import defaultdict
import cloudpickle
import pickle
import pdb

class ReplayMemory:
    def __init__(self, capacity,
                 sample_data_fn=lambda x: x,
                 push_data_fn=lambda x: [x]):
        self.capacity = capacity
        self.sample_data_fn = sample_data_fn
        self.push_data_fn = push_data_fn
        self.memory = [None] * capacity
        self.pos = 0
        self.length = 0
        self._key_vars = ['pos', 'length', 'memory']
    
    def reset(self):
        self.pos = 0
        self.length = 0

    def __push(self, item):
        self.memory[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def push(self, item, fn_call=False):
        if fn_call:
            for _item in self.push_data_fn(item):
                self.__push(_item)
        else:
            self.__push(item)

    def sample(self, batch_size):
        batch += random.sample(self.memory[:self.length],
                               min(batch_size, self.length))
        return self.sample_data_fn(batch)

    def __len__(self):
        return self.length

    def save(self, filename):
        data = [getattr(self, var) for var in self._key_vars]
        with open(filename, 'wb') as fw:
            cloudpickle.dump(data, fw)

    def load(self, filename):
        with open(filename, 'rb') as fr:
            data = pickle.load(fr)
        for name, value in zip(self._key_vars, data):
            setattr(self, name, value)


class PrioritizedReplayMemory(ReplayMemory):
    #epsilon: float = 0.01 # small amount to avoid zero priority
    #alpha: float = 0.6 # [0~1] convert the importance of TD error to priority
    #beta: float = 0.4 # importance-sampling, from initial value increasing to 1
    #abs_err_upper: float = 1.  # clipped abs error
    #beta_increment_per_sampling: float = 0.001
    
    def __init__(self, capacity, alpha=0.6, epsilon=1e-6,
                 sample_data_fn=lambda x: x,
                 push_data_fn=lambda x: [x]):
        super(PrioritizedReplayMemory, self).__init__(
            capacity,
            sample_data_fn=sample_data_fn,
            push_data_fn=push_data_fn
        )
        # sum_tree
        self.tree = [0.] * (2 * capacity - 1)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0
        self._key_vars += ['tree', 'max_priority']

    def reset(self):
        super().reset()
        self.tree = [0.] * (2 * self.capacity - 1)

    def push(self, item):
        for _item in self.push_data_fn(item):
            #pdb.set_trace()
            idx = self.pos + self.capacity - 1
            super().push(_item, False)
            priority = self.max_priority ** self.alpha
            self.update_sumtree(idx, priority, is_error=False)

    def sample(self, batch_size, beta):
        idxs, batch, isweights = [], [], []
        segment = self.tree[0] / batch_size
        
        if self.length != self.capacity:
            min_prob = np.min(self.tree[-self.capacity:-self.capacity + self.length]) / self.tree[0]
        else:
            min_prob = np.min(self.tree[-self.capacity:]) / self.tree[0]
        assert min_prob != 0
        max_weight = (min_prob * self.length) ** (-beta)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority = self.get_from_sumtree(s)

            if self.memory[idx + 1 - self.capacity] is None: continue

            weight = (priority * self.length) ** (-beta)
            isweights.append(weight / max_weight)
            batch.append(self.memory[idx + 1 - self.capacity])
            idxs.append(idx)

        return idxs, torch.tensor(isweights), self.sample_data_fn(batch)

    def update_sumtree(self, idx, value, is_error=True):
        priority = self._get_priority(value) if is_error else value
        self.max_priority = max(self.max_priority, priority)

        change = priority - self.tree[idx]
        self.tree[idx] = priority

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def batch_update_sumtree(self, batch_idx, batch_value, is_error=True):
        for idx, value in zip(batch_idx, batch_value):
            if idx == -1: continue
            self.update_sumtree(idx, value, is_error)

    def get_from_sumtree(self, x):
        cur = 0
        while 2 * cur + 1 < len(self.tree):
            left = 2 * cur + 1
            right = left + 1
            if self.tree[left] >= x:
                cur = left
            else:
                x -= self.tree[left]
                cur = right
        return cur, self.tree[cur]

    def _get_priority(self, error):
        #return min(error + self.epsilon, self.abs_err_upper) ** self.alpha
        return (error + self.epsilon) ** self.alpha


