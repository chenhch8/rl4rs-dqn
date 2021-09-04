#!/usr/bin/env python
# coding=utf-8
import numpy as np
from prettytable import PrettyTable

def table_format(data, field_names=None, title=None):
    tb = PrettyTable()
    if field_names is not None:
        tb.field_names = field_names
        for i, name in enumerate(field_names):
            tb.align[name] = 'r' if i else 'l'
    if title is not None:
        tb.title = title
    tb.add_rows(data)
    return tb.get_string()


def recall(batch_pred_bundles, batch_target_bundles):
    rec, rec1, rec2, rec3 = [], [], [], []
    for pred_bundle, target_bundle in zip(batch_pred_bundles, batch_target_bundles):
        recs = []
        for bundle_a, bundle_b in zip(pred_bundle, target_bundle):
            recs.append(len(set(bundle_a.tolist()) & set(bundle_b.tolist())) / len(bundle_b))
        rec1.append(recs[0])
        rec2.append(recs[1])
        rec3.append(recs[2])
        rec.append((rec1[-1] + rec2[-1] + rec3[-1]) / 3)
    return np.mean(rec), np.mean(rec1), np.mean(rec2), np.mean(rec3)


def nan2num(tensor, num=0):
    tensor[tensor != tensor] = num


def inf2num(tensor, num=0):
    tensor[tensor == float('-inf')] = num
    tensor[tensor == float('inf')] = num


def tensor2device(tensors, device):
    return [tensor.to(device) if tensor is not None else None \
            for tensor in tensors]

