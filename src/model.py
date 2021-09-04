#!/usr/bin/env python
# coding=utf-8
import torch
import math
from torch import nn
# from torch.nn import TransformerEncoder, \
#                      TransformerEncoderLayer, \
#                      init
import numpy as np
from tools import nan2num, inf2num

import pdb


def init_weights(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )
        self.apply(init_weights)

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        query: [batch, seq1, hidden_size]
        key: [batch, seq2, hidden_size * num_hidden_state]
        value: [batch, seq2, hidden_size * num_hidden_state]
        q_mask: [batch, seq1]
        k_mask: [batch, seq2]

        return:
            [batch, seq1, hidden_size]
        '''
        batch, seq1, hidden_size1 = query.size()
        _, seq2, hidden_size2 = key.size()

        if q_mask.dtype != torch.float:
            q_mask = q_mask.type(torch.float)
        if k_mask.dtype != torch.float:
            k_mask = k_mask.type(torch.float)

        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, seq1, seq2))
        
        query_e = query.unsqueeze(2).expand(-1, -1, seq2, -1)
        key_e = key.unsqueeze(1).expand(-1, seq1, -1, -1)
        stack = torch.cat([query_e, key_e], dim=-1)
        assert stack.size() == torch.Size((batch, seq1, seq2, hidden_size1 + hidden_size2))
        # [batch, seq1, seq2]
        A = self.mlp(stack) \
                .squeeze(-1) \
                .masked_fill(mask == 0, float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=1)
        attn = A.div(A_sum)
        assert A.size() == torch.Size((batch, seq1, seq2))
        return attn.matmul(value)



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, q_mask, k_mask, scale=None):
        '''
        q: [B, L_q, D_q]
        k: [B, L_k, D_k]
        v: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
        batch, L_q, D_q = query.size()
        _, L_k, D_k = key.size()

        if scale is None:
            scale = D_q
        
        if q_mask.dtype != torch.float:
            q_mask = q_mask.type(torch.float)
        if k_mask.dtype != torch.float:
            k_mask = k_mask.type(torch.float)

        #print(q_mask.size(), k_mask.size())
        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, L_q, L_k))

        # [batch, L_q, L_k]
        A = query.matmul(key.transpose(1, 2)) \
                .div(np.sqrt(scale)) \
                .masked_fill(mask == 0, float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=1.)
        attn = A.div(A_sum)
        assert attn.size() == torch.Size((batch, L_q, L_k))
        return attn.matmul(value)


class MLPLayer(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None):
        super(MLPLayer, self).__init__()
        if hidden_size is None:
            hidden_size = in_size
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, out_size),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, nheads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim_head = dim // nheads
        self.nheads = nheads
        self.linear_k = nn.Linear(dim, self.dim_head * nheads)
        self.linear_v = nn.Linear(dim, self.dim_head * nheads)
        self.linear_q = nn.Linear(dim, self.dim_head * nheads)

        self.dot_product_attn = ScaledDotProductAttention()
        self.linear_final = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(dim)
        self.apply(init_weights)

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        query: [B, L_q, D_q]
        q_mask: [B, L_q]
        k_mask: [k_q]
        '''
        residual = query
        batch = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch * self.nheads, -1, self.dim_head)
        value = value.view(batch * self.nheads, -1, self.dim_head)
        query = query.view(batch * self.nheads, -1, self.dim_head)

        q_mask = q_mask.repeat(self.nheads, 1)
        k_mask = k_mask.repeat(self.nheads, 1)

        context = self.dot_product_attn(query=query,
                                        key=key,
                                        value=value,
                                        q_mask=q_mask,
                                        k_mask=k_mask,
                                        scale=self.dim_head)
        context = context.view(batch, -1, self.nheads * self.dim_head)
        
        output = self.linear_final(context)
        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, dim=512, ffn_dim=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        #self.w1 = nn.Conv1d(dim, ffn_dim, 1)
        #self.w2 = nn.Conv1d(dim, ffn_dim, 1)
        self.fc = MLPLayer(in_size=dim, hidden_size=ffn_dim, out_size=dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.apply(init_weights)

    def forward(self, x):
        '''
        x: [B, S, D]
        '''
        #output = x.transpose(1, 2)
        #output = self.w2(torch.relu(self.w1(output)))
        #output = self.dropout(output.transpose(1, 2))
        output = self.dropout(self.fc(x))

        return self.layer_norm(x + output)

class Transformer(nn.Module):
    def __init__(self, dim, nheads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.nheads = nheads
        self.attention = MultiHeadAttention(dim=dim, nheads=nheads, dropout=dropout)
        dim = (dim // nheads) * nheads
        self.pos_fc = PositionalWiseFeedForward(dim=dim, dropout=dropout, ffn_dim=2 * dim)
    
    #def forward(self, query, q_mask, key=None, value=None, k_mask=None):
    def forward(self, data):
        '''
        query: [B, L_q, D_q]
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
        if len(data) == 5:
            query, q_mask, key, value, k_mask = data
        elif len(data) == 2:
            query, q_mask = data
            key = query
            value = query
            k_mask = q_mask
        else:
            raise ValueError

        B, L_q, dim = query.size()
        L_k = key.size(1)

        output = self.attention(
            query=query,
            key=key,
            value=value,
            q_mask=q_mask,
            k_mask=k_mask
        )
        dim = (dim // self.nheads) * self.nheads
        assert output.size() == torch.Size((B, L_q, dim))
        output = self.pos_fc(output)
        return (output, q_mask, key, value, k_mask)


class PositionEmbedding(nn.Module):
    def __init__(self, pos_size, pos_dim, dropout=0.1):
        super(PositionEmbedding, self).__init__()

        pos_embed = torch.zeros(pos_size, pos_dim, requires_grad=True)
        position = torch.arange(0, pos_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2, dtype=torch.float) * \
                             (-math.log(10000.0) / pos_dim)).unsqueeze(0)
        pos_embed[:, 0::2] = torch.sin(position * div_term)[:, :pos_dim // 2 + 1]
        pos_embed[:, 1::2] = torch.cos(position * div_term)[:, :pos_dim // 2]

        #self.pos_embed = nn.Parameter(pos_embed)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        '''
        x: [batch, seq, dim]
        '''
        return self.dropout(x + self.pos_embed[None, :x.size(1), :])


class QNetwork(nn.Module):
    def __init__(self,
                 item_embed,
                 out_dim=381,
                 num_layers=[3, 3],
                 user_dim=10,
                 bundle_size=3,
                 nhead=8,
                 seq_length=356,
                 dropout=0.1,
                 dueling=True):
        super(QNetwork, self).__init__()
        
        hidden_size = item_embed.size(1)
        
        self.item_embed = item_embed
        self.out_dim = out_dim
        self.dueling = dueling
        self.encoder_pos_embed = PositionEmbedding(seq_length,
                                                   hidden_size,
                                                   dropout=dropout)
        self.decoder_pos_embed = PositionEmbedding(bundle_size + 1,
                                                   bundle_size * hidden_size + user_dim,
                                                   dropout=dropout)
        
        self.init_fc = nn.Linear(hidden_size, bundle_size * hidden_size)
        self.ACT = nn.Parameter(torch.zeros(1, 1, bundle_size * hidden_size))
        self.attn_history2user = AttentionLayer(hidden_size + user_dim, 2 * (hidden_size + user_dim))
        self.attn_history2item = AttentionLayer(2 * hidden_size, 4 * hidden_size)
        # self.history_encoder = TransformerEncoder(
        #     TransformerEncoderLayer(d_model=hidden_size,
        #                             nhead=nhead,
        #                             dim_feedforward=2 * hidden_size,
        #                             dropout=dropout),
        #     num_layers[0]
        # )
        self.history_encoder = nn.Sequential(
            *[Transformer(hidden_size,
                          nheads=nhead,
                          dropout=dropout) for _ in range(num_layers[0])]
        )
        # self.bundle_decoder = TransformerEncoder(
        #     TransformerEncoderLayer(d_model=bundle_size * hidden_size + user_dim,
        #                             nhead=nhead,
        #                             dim_feedforward=2 * (bundle_size * hidden_size + user_dim),
        #                             dropout=dropout),
        #     num_layers[1]
        # )
        self.bundle_decoder = nn.Sequential(
            *[Transformer(bundle_size * hidden_size + user_dim,
                          nheads=nhead,
                          dropout=dropout) for _ in range(num_layers[1])]
        )
        self.predict = nn.Linear(
            bundle_size * hidden_size + user_dim,
            out_dim,
            bias=True
        )
        if self.dueling:
            self.val = nn.Sequential(
                nn.Linear(2 * (bundle_size * hidden_size + user_dim),
                          bundle_size * hidden_size + user_dim),
                nn.ReLU(),
                nn.Linear(bundle_size * hidden_size + user_dim, 1)
            ) 

    def forward(self,
                batch_users,
                batch_encoder_item_ids,
                encoder_mask,
                batch_decoder_item_ids=None,
                decoder_mask=None,
                act_mask=None):
        '''
        batch_users: [batch, user_dim]
        batch_encoder_item_ids: [batch, seq]
        encoder_mask: [batch, seq]
        batch_decoder_item_ids: [batch, seq2, 3], where seq2 \in [0, 2]
        decoder_mask: [batch, seq2]
        act_mask: [batch, out_dim]

        return: [batch, out_dim]
        '''
        if self.item_embed.device != self.init_fc.weight.device:
            self.item_embed = self.item_embed.to(self.init_fc.weight.device)
        # [batch, seq, item_dim]
        batch_encoder_items = self.item_embed[batch_encoder_item_ids] * encoder_mask.unsqueeze(2)
        batch, seq = batch_encoder_item_ids.size()
        seq2 = batch_decoder_item_ids.size(1) if batch_decoder_item_ids is not None else 0
        item_dim = batch_encoder_items.size(-1)
        user_dim = batch_users.size(1)
        # [batch, seq, item_dim]
        # batch_encoder_items = self.history_encoder(
        #     self.encoder_pos_embed(batch_encoder_items).transpose(0, 1),
        #     #src_key_padding_mask=~encoder_mask
        #     src_key_padding_mask=encoder_mask == False
        # ).transpose(0, 1)
        # pdb.set_trace()
        #print('encoder_mask', encoder_mask.size())
        batch_encoder_items = self.history_encoder(
            (self.encoder_pos_embed(batch_encoder_items) * encoder_mask.unsqueeze(2),
             encoder_mask)
        )[0] * encoder_mask.unsqueeze(2)
        if torch.isnan(batch_encoder_items).sum() != 0:
            pdb.set_trace()
        #nan2num(batch_encoder_items, 0)
        assert batch_encoder_items.size() == torch.Size((batch, seq, item_dim))

        # [batch, 1, item_dim]
        user_items = self.attn_history2user(
            query=batch_users.unsqueeze(1),
            key=batch_encoder_items,
            value=batch_encoder_items,
            q_mask=encoder_mask.new_ones(batch, 1),
            k_mask=encoder_mask,
        )
        assert user_items.size() == torch.Size((batch, 1, item_dim))
        user_items = self.init_fc(user_items) * \
                (encoder_mask.sum(1, keepdim=True) != 0).unsqueeze(1)
        assert user_items.size() == torch.Size((batch, 1, 3 * item_dim))
        ACT = self.ACT + user_items
        
        decoder_items = None
        if batch_decoder_item_ids is not None:
            decoder_items = self.item_embed[batch_decoder_item_ids] * decoder_mask[:, :, None, None]
            assert decoder_items.size() == torch.Size((batch, seq2, 3, item_dim))
            item2history = self.attn_history2item(
                query=decoder_items.view(batch, -1, item_dim),
                key=batch_encoder_items,
                value=batch_encoder_items,
                q_mask=decoder_mask.unsqueeze(-1).expand(-1, -1, 3).contiguous().view(batch, -1),
                k_mask=encoder_mask
            )
            assert item2history.size() == torch.Size((batch, 3 * seq2, item_dim))
            decoder_items = decoder_items + item2history.view(batch, seq2, 3, item_dim)
        decoder_items = torch.cat([ACT, decoder_items.view(batch, seq2, -1)],
                                  dim=1) \
                if decoder_items is not None else ACT
        decoder_items = torch.cat([decoder_items,
                                   batch_users.unsqueeze(1).expand(-1, seq2 + 1, -1)],
                                  dim=-1)
        assert decoder_items.size() == torch.Size((batch, seq2 + 1, 3 * item_dim + user_dim))

        #src_mask = torch.arange(seq2 + 1).unsqueeze(0).expand(seq2 + 1, -1) <= torch.arange(seq2 + 1).unsqueeze(1)
        src_key_padding_mask = torch.cat([decoder_mask.new_ones(batch, 1), decoder_mask], dim=1) \
                if decoder_mask is not None else encoder_mask.new_ones(batch, 1)
        #pdb.set_trace()
        # logits = self.bundle_decoder(
        #     self.decoder_pos_embed(decoder_items).transpose(0, 1),
        #     #src_mask == False,
        #     src_key_padding_mask=src_key_padding_mask == False
        # ).transpose(0, 1)
        logits = self.bundle_decoder(
            (self.decoder_pos_embed(decoder_items) * src_key_padding_mask.unsqueeze(2),
             src_key_padding_mask)
        )[0] * src_key_padding_mask.unsqueeze(2)
        action_scores = self.predict(logits[:, 0, :].squeeze(1))
        if torch.isnan(action_scores).sum() != 0:
            pdb.set_trace()
        assert action_scores.size() == torch.Size((batch, self.out_dim))
        
        if self.dueling:
            _act_mask = (act_mask == 1) if decoder_mask is None else \
                    act_mask == decoder_mask.sum(dim=1, keepdim=True) + 1
            #action_scores = action_scores - action_scores.mean(dim=1, keepdim=True)
            action_scores = action_scores - \
                    (action_scores * _act_mask).sum(dim=1, keepdim=True) / _act_mask.sum(dim=1, keepdim=True).clamp(min=1)
            if logits.size(1) > 1:
                _max_agg = logits[:, 1:, :].masked_fill(decoder_mask.unsqueeze(2) == False,
                                                        float('-inf')) \
                                           .max(dim=1)[0]
                _avg_agg = (logits[:, 1:, :] * decoder_mask.unsqueeze(2)).sum(dim=1) \
                                           .div(decoder_mask.sum(dim=1, keepdim=True).clamp(min=1))
                inf2num(_max_agg)
                state_scores = self.val(torch.cat([
                    _max_agg, _avg_agg
                ], dim=-1))
                assert state_scores.size() == torch.Size((batch, 1))
            else:
                state_scores = 0
            scores = state_scores + action_scores
        else:
            scores = action_scores

        return scores


if __name__ == '__main__':
    out_dim = 9
    num_layers = [3, 3]
    user_dim = 10
    bundle_size = 3
    nhead = 2
    seq_length = 15
    dropout = 0.1
    item_embed = torch.rand(seq_length, 10)
    
    batch = 5
    batch_users = torch.rand(batch, user_dim)
    batch_encoder_item_ids = torch.tensor(np.random.randint(0, seq_length, (batch, 5)),
                                          dtype=torch.long)
    encoder_mask = torch.arange(5).unsqueeze(0).expand(batch, -1) < torch.randint(0, 6, (batch, 1))
    batch_decoder_item_ids = torch.tensor([[[1, 0, 0], [1, 2, 3]],
                                           [[0, 0, 0], [0, 0, 0]],
                                           [[0, 1, 5], [0, 0, 0]],
                                           [[0, 0, 0], [0, 0, 0]],
                                           [[2, 0, 0], [1, 0, 2]]], dtype=torch.long)
    decoder_mask = torch.tensor([[1, 1],
                                 [0, 0],
                                 [1, 0],
                                 [0, 0],
                                 [1, 1]], dtype=torch.bool)

    qnet = QNetwork(item_embed,
                    out_dim,
                    num_layers,
                    user_dim,
                    bundle_size,
                    nhead,
                    seq_length,
                    dropout)
    scores = qnet(
        batch_users,
        batch_encoder_item_ids,
        encoder_mask,
        batch_decoder_item_ids,
        decoder_mask
    )
    print(scores)

    scores2 = qnet(
        batch_users,
        batch_encoder_item_ids,
        encoder_mask,
    )
    print(scores2)
