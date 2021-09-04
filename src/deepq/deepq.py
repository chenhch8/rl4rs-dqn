#!/usr/bin/env python
# coding=utf-8
import random
import torch
import numpy as np
import os

import deepq
from model import QNetwork
from replay_buffer import PrioritizedReplayMemory
from deepq.schedules import LinearSchedule
from tools import table_format, recall
import pdb

def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sample_data_stack(batch_data):
    def _obs(batch_obs):
        batch_obs = [obs for obs in batch_obs if obs is not None]
        batch_users, batch_click_items, batch_click_masks, \
                batch_exposed_items, _ = list(zip(*batch_obs))
            
        lengths = [item.size(0) if item is not None else 0 for item in batch_exposed_items]
        max_length = max(lengths)
        batch_exposed_mask = torch.tensor([[1] * length + [0] * (max_length - length) \
                                           for length in lengths])
        _batch_exposed_items = batch_exposed_mask[0].new_zeros((len(batch_exposed_items), max_length, 3))
        for i in range(len(batch_exposed_items)):
            if batch_exposed_items[i] is None: continue
            _batch_exposed_items[i, :lengths[i]] = batch_exposed_items[i].squeeze(0)
        batch_exposed_items = _batch_exposed_items
    
        return torch.stack(batch_users), torch.stack(batch_click_items), \
                torch.stack(batch_click_masks), batch_exposed_items, batch_exposed_mask 
            
    batch_obs, batch_acts, batch_rews, batch_new_obs, \
            batch_done, batch_exposed_items = list(zip(*batch_data))
    batch_obs, batch_new_obs = _obs(batch_obs), _obs(batch_new_obs)
    
    return batch_obs, torch.stack(batch_acts), torch.stack(batch_rews), \
            batch_new_obs, torch.tensor(batch_done), torch.stack(batch_exposed_items)


def push_data_unstack(batch_data):
    def _obs(batch_obs):
        batch_obs = list(batch_obs)
        if batch_obs[0] is not None:
            if batch_obs[3] is None:
                batch_obs[3] = [None] * batch_obs[0].size(0)
                batch_obs[4] = [None] * batch_obs[0].size(0)
            return list(zip(*batch_obs))
        else:
            return batch_obs
            
    batch_obs, batch_acts, batch_rews, batch_new_obs, \
            batch_done, batch_exposed_items = batch_data
    batch_obs, batch_new_obs = _obs(batch_obs), _obs(batch_new_obs)

    return list(zip(batch_obs, batch_acts, batch_rews, batch_new_obs, batch_done, batch_exposed_items))

    
def learn(env,
          network,
          trainset,
          act_mask,
          seed=42,
          lr=5e-4,
          total_epochs=10000,
          buffer_size=50000,
          exploration_fraction=0.9,
          exploration_final_eps=0.01,
          batch_size=128,
          print_freq=2,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=500,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_eps=1e-6,
          topk=3,
          T=3):
    
    set_global_seeds(seed)
    
    q_net, act, train, update_target, save_model, load_model = deepq.build_train(
        q_net=QNetwork(**network),
        optimizer=lambda params: torch.optim.Adam(params,
                                                  lr=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        is_gpu=torch.cuda.is_available(),
        act_mask=act_mask
    )

    replay_buffer = PrioritizedReplayMemory(
        buffer_size, prioritized_replay_alpha, prioritized_replay_eps,
        sample_data_fn=sample_data_stack,
        push_data_fn=push_data_unstack,
    )
    beta_schedule = LinearSchedule(
        total_epochs * len(trainset), initial_p=prioritized_replay_beta0,
        final_p=1.0
    )
    exploration = LinearSchedule(
        int(total_epochs * len(trainset) * exploration_fraction),
        initial_p=1.0,
        final_p=exploration_final_eps
    )
    
    update_target()

    episode_rewards = [0.0]
    episode_greedy_eps = []
    saved_mean_reward = None
    start_steps = 0
    
    if os.path.exists(os.path.join(checkpoint_path, 'best_checkpoint.pt')):
        print(f'loading best checkpoint from {checkpoint_path} ...')
        start_steps, episode_rewards, saved_mean_reward = \
                load_model(os.path.join(checkpoint_path, 'best_checkpoint.pt'))
        episode_rewards, saved_mean_reward = [0.0], None
        #replay_buffer.load(os.path.join(checkpoint_path, 'replay_buffer.pk'))

    def interact_with_env(init_obs, is_greedy=False, target_bundles=None):
        if target_bundles is not None:
            target_bundles = target_bundles.view(target_bundles.size(0), 3, 3)
            #if torch.cuda.is_available():
            #    target_bundles = target_bundles.to(torch.device('cuda'))
        obs, pre_obs = init_obs, None
        for t in range(T):
            act_, _greedy_eps = act(obs,
                                    #(act_mask == t + 1).unsqueeze(0).expand(obs[0].size(0), -1),
                                    exploration.value(steps),
                                    is_greedy=is_greedy)
            act_ = act_.cpu()
            if not is_greedy:
                new_obs, rew, done = env.step(obs, act_, target_bundles, t)
                replay_buffer.push((obs, act_, rew, new_obs, done, target_bundles))
                episode_rewards[-1] += rew.mean().item()
                if done[0]:
                    episode_rewards.append(0.0)
                    episode_greedy_eps.append(_greedy_eps)
            else:
                new_obs = env.new_obs(obs, act_)
            pre_obs, obs = obs, new_obs
        return torch.cat([pre_obs[3], act_.unsqueeze(1)], dim=1)
    
    cached_data = []
    steps = 0
    for _ in range(total_epochs):
        trainset.reset()
        for batch_user, batch_click_items, batch_click_mask, batch_exposed_items in trainset:
            if len(cached_data) < 5:
                cached_data.append((batch_user, batch_click_items, batch_click_mask, batch_exposed_items))
            if steps < start_steps:
                steps += 1
                continue
            interact_with_env((batch_user, batch_click_items, batch_click_mask, None, None),
                              target_bundles=batch_exposed_items)
            # train
            #if steps > learning_starts and len(replay_buffer) >= batch_size:
            if len(replay_buffer) >= batch_size:
                idxs, isweights, batch_data = replay_buffer.sample(batch_size,
                                                                   beta=beta_schedule.value(steps))
                batch_obs, batch_acts, batch_rews, batch_new_obs, batch_done, _ = batch_data
                wloss, loss, error = train(batch_obs, batch_acts, batch_rews, batch_new_obs,
                                           isweights, batch_done, topk)
                replay_buffer.batch_update_sumtree(idxs, error, is_error=True)
                if steps % target_network_update_freq == 0:
                    update_target()
                
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 3)
                mean_100ep_greedy_eps = round(np.mean(episode_greedy_eps[-100:]), 3)
                num_episodes = len(episode_rewards)
                
                #print(len(episode_rewards), print_freq, wloss, loss)
                if print_freq is not None and len(episode_rewards) % print_freq == 0:
                    batch_users, batch_click_items, batch_click_mask, batch_exposed_items \
                            = [torch.cat(data, dim=0) for data in zip(*cached_data)]
                    selected_bundles = interact_with_env(
                        (batch_users, batch_click_items, batch_click_mask, None, None),
                        is_greedy=True
                    )
                    rec, rec1, rec2, rec3 = recall(selected_bundles,
                                                   batch_exposed_items.view(selected_bundles.size()))
                    print(table_format(
                        [['progress', round(steps / (total_epochs * len(trainset)), 5)],
                         ['episodes', num_episodes],
                         ['mean 100 episode reward', round(mean_100ep_reward, 5)],
                         ['mean 100 episode greedy eps', round(mean_100ep_greedy_eps, 5)],
                         ['prioritized replay beta', round(beta_schedule.value(steps), 5)],
                         ['wloss', round(wloss, 5)],
                         ['loss', round(loss, 5)],
                         ['recall', round(rec, 5)],
                         ['recall-1', round(rec1, 5)],
                         ['recall-2', round(rec2, 5)],
                         ['recall-3', round(rec3, 5)]],
                        field_names=['steps', steps])
                    )

                #if (num_episodes > 100 and steps % checkpoint_freq == 0) and \
                if num_episodes > 100 and \
                   (saved_mean_reward is None or mean_100ep_reward > saved_mean_reward):
                    saved_mean_reward = mean_100ep_reward
                    save_model(os.path.join(checkpoint_path, 'best_checkpoint.pt'),
                               steps, episode_rewards, saved_mean_reward)
                    #replay_buffer.save(os.path.join(checkpoint_path, 'replay_bufer.pk'))
            steps += 1

    return act
