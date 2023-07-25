"""
This script contains the implementation of a GLFormer.

The script is composed of several classes and a helper function:

1. `LScaledDotProductAttention`: This class defines a lightweight scaled dot product attention mechanism, which computes the attention weights and applies them to the input values with the group mechanism.

2. `LMultiHeadAttention`: This class represents the lightweight multi-head attention, which applies several `LScaledDotProductAttention` modules to the input and combines their results.

3. `Lightformer`: This class is the lightweight GLFormer that uses the custom attention mechanism defined by `LMultiHeadAttention`. The model passes the input through each layer of the GLFormer.

4. `LightformerLayer`: This class represents a single layer of the GLFormer (Global-Former or Local-Former, depending on the input mask), which includes a lightweight self-attention mechanism and a lightweight feed-forward neural network, both of them employ the group mechanism.

5. `_get_clones`: This helper function creates multiple copies of a given module, which is useful for creating several identical layers in a neural network.

"""

import copy, torch
from typing import Optional
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import *
from torch.nn.init import constant_, xavier_uniform_
import numpy as np


# Define a Scaled Dot Product Attention to compute Q, K, V in GLformers
class LScaledDotProductAttention(Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, groups=2):
        super(LScaledDotProductAttention, self).__init__()

        self.fc_q = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_k = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_v = nn.Linear(d_model // groups, h * d_v // groups)
        self.fc_o = nn.Linear(h * d_v // groups, d_model // groups)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.fc_q.weight)
        xavier_uniform_(self.fc_k.weight)
        xavier_uniform_(self.fc_v.weight)
        xavier_uniform_(self.fc_o.weight)
        constant_(self.fc_q.bias, 0)
        constant_(self.fc_k.bias, 0)
        constant_(self.fc_v.bias, 0)
        constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries.view(b_s, nq, self.groups, -1)).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out.view(b_s, nq, self.groups, -1)).view(b_s, nq, -1)
        return out.permute(1, 0, 2)


# Define the Multi-Head Attention, which uses the Scaled Dot Product Attention
class LMultiHeadAttention(Module):
    def __init__(self, d_model, h, dropout=.1, groups=2):
        super(LMultiHeadAttention, self).__init__()

        self.attention = LScaledDotProductAttention(d_model=d_model, groups=groups, d_k=d_model // h, d_v=d_model // h,
                                                    h=h, dropout=dropout)

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None, need_weights=False,
                attention_weights=None):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out


# Define the lightweight GLFormer module, which consists of several GLFormers (i.e. LightformerLayer).
class Lightformer(Module):
    __constants__ = ['norm']

    def __init__(self, attention_layer, num_layers, norm=None):
        super(Lightformer, self).__init__()
        self.layers = _get_clones(attention_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for i, mod in enumerate(self.layers):
            if i % 2 == 0:
                output = mod(output)
            else:
                output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


# Define a single GLFormer layer, which uses the Multi-Head Attention
class LightformerLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super(LightformerLayer, self).__init__()
        self.self_attn = LMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward // 2, d_model // 2)  ###

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            self.activation = F.gelu
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(LightformerLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # Define the Self-Attention module
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # Define the Feed-Forward Network module (FFN)
    def _ff_block(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        x = self.linear2(self.dropout(self.activation(self.linear1(x))).view(b, l, 2, d * 4 // 2))  ###
        x = x.view(b, l, d)
        return self.dropout2(x)


# Helper function to create multiple copies of a module
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

