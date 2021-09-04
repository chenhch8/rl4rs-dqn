#!/usr/bin/env python
# coding=utf-8
import math
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict

from tools import tensor2device

import pdb

def _calc_q_value(obs, net, act_mask, device):
    batch_users, batch_encoder_item_ids, encoder_mask, \
            batch_decoder_item_ids, decoder_mask = tensor2device(obs, device)
    return net(batch_users,
               batch_encoder_item_ids,
               encoder_mask,
               batch_decoder_item_ids,
               decoder_mask,
               act_mask.unsqueeze(0).expand(batch_users.size(0), -1) if act_mask is not None else None)


def build_train(q_net,
                optimizer,
                grad_norm_clipping,
                act_mask,
                gamma=0.99,
                is_gpu=False):
    device = torch.device('cuda') if is_gpu else torch.device('cpu')
    q_net.to(device)

    t_net = deepcopy(q_net)
    t_net.eval()
    t_net.to(device)
    optim = optimizer(q_net.parameters())

    act_mask = act_mask.to(device)

    if is_gpu and torch.cuda.device_count() > 1:
        q_net = torch.nn.DataParallel(q_net)
        t_net = torch.nn.DataParallel(t_net)

    def save_model(filename,
                   epoch,
                   episode_rewards,
                   saved_mean_reward):
        torch.save({
            'epoch': epoch,
            'episode_rewards': episode_rewards,
            'saved_mean_reward': saved_mean_reward,
            'model': q_net.state_dict(),
            'optim': optim.state_dict()
        }, filename)

    def load_model(filename):
        checkpoint = torch.load(filename,
                                map_location=torch.device('cpu'))
        #q_net.load_state_dict(checkpoint['model'])
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k.find('module.') != -1:
                k = k[7:]
            new_state_dict[k] = v
        q_net.load_state_dict(new_state_dict)
        optim.load_state_dict(checkpoint['optim'])
        return checkpoint['epoch'], checkpoint['episode_rewards'], checkpoint['saved_mean_reward']

    def train(obs,
              act,
              rew,
              next_obs,
              isweights,
              done_mask,
              topk=3):
        act, rew, isweights = act.to(device), rew.to(device), isweights.to(device)
        # q value at t+1 in double q
        with torch.no_grad():
            q_net.eval()
            next_q_val = _calc_q_value(next_obs, q_net, act_mask, device).detach()
            q_net.train()
            
            _next_mask = next_obs[4].to(device).sum(dim=1, keepdim=True) + 1 == act_mask.unsqueeze(0)
            assert next_q_val.size() == _next_mask.size()

            next_q_val[_next_mask == False] = float('-inf')

            next_action_max = next_q_val.argsort(dim=1, descending=True)[:, :topk]
            next_q_val_max = _calc_q_value(next_obs, t_net, act_mask, device) \
                                   .detach() \
                                   .gather(dim=1, index=next_action_max) \
                                   .sum(dim=1)

            _next_q_val_max = next_q_val_max.new_zeros(done_mask.size())
            _next_q_val_max[done_mask == False] = next_q_val_max
        # q value at t
        q_val = _calc_q_value(obs, q_net, act_mask, device)
        q_val_t = q_val.gather(dim=1, index=act.to(device)).sum(dim=1)
        assert q_val_t.size() == _next_q_val_max.size()
        #print('done')
        # Huber Loss
        loss = F.smooth_l1_loss(q_val_t,
                                rew + gamma * _next_q_val_max,
                                reduction='none')
        assert loss.size() == isweights.size()
        #wloss = (loss * isweights).mean()
        wloss = loss.mean()
        wloss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_norm_clipping)
        optim.step()
        q_net.zero_grad()

        return wloss.detach().data.item(), (loss.detach().mean().data.item()), loss.cpu().detach().abs()

    def act(obs,
            eps_greedy,
            topk=3,
            is_greedy=False):
        return build_act(obs, act_mask, q_net, eps_greedy, topk,
                         is_greedy=is_greedy, device=device)

    def update_target():
        for target_param, local_param in zip(t_net.parameters(), q_net.parameters()):
            target_param.data.copy_(local_param.data)

    return q_net, act, train, update_target, save_model, load_model


def build_act(obs,
              act_mask,
              net,
              eps_greedy,
              topk=3,
              is_greedy=False,
              device=None):
    devcie = torch.device('cpu') if device is None else device
    act_mask = act_mask.to(device)
    def _epsilon_greedy(size):
        return torch.rand(size).to(device) < eps_greedy
    def _gen_act_mask():
        #if obs[3] is not None:
        if obs[4] is not None:
            #length = torch.tensor([len(o) + 1 if o is not None else 1 for o in obs[3]],
            #                      dtype=torch.float).view(-1, 1).to(device)
            length = obs[4].to(device).sum(dim=1, keepdim=True) + 1
        else:
            length = act_mask.new_ones((1,)).view(-1, 1)
        return act_mask.unsqueeze(0) == length
    net.eval()
    with torch.no_grad():
        q_val = _calc_q_value(obs, net, act_mask, device).detach()
        _act_mask = _gen_act_mask()
        if q_val.size() != _act_mask.size():
            assert _act_mask.size(0) == 1
            _act_mask = _act_mask.expand(q_val.size(0), -1)
        q_val[_act_mask == False] = float('-inf')
        _deterministic_acts = q_val.argsort(dim=1, descending=True)[:, :topk]
        if not is_greedy:
            _stochastic_acts = _deterministic_acts.new_empty(_deterministic_acts.size())
            chose_random = _epsilon_greedy(_stochastic_acts.size(0))
            _tmp = torch.arange(0, _act_mask.size(1), dtype=_deterministic_acts.dtype)
            for i in range(_act_mask.size(0)):
                _available_acts = _act_mask[i].nonzero().view(-1)
                _stochastic_acts[i] = _available_acts[torch.randperm(_available_acts.size(0))[:topk]]
            #if chose_random.sum() != len(chose_random):
            #    pdb.set_trace()
            _acts = torch.where(chose_random.unsqueeze(1).expand(-1, _stochastic_acts.size(1)),
                                _stochastic_acts,
                                _deterministic_acts)
            # TODO 去重       
        else:
            _acts = _deterministic_acts
            eps_greedy = 0.
    net.train()
    
    return _acts, eps_greedy
