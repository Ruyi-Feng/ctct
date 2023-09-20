
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, args):
        super().__init__()
        assert args.d_model % args.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(args.d_model, args.d_model)
        self.query = nn.Linear(args.d_model, args.d_model)
        self.value = nn.Linear(args.d_model, args.d_model)
        # regularization
        self.attn_drop = nn.Dropout(args.attn_pdrop)
        self.resid_drop = nn.Dropout(args.resid_pdrop)
        # output projection
        self.proj = nn.Linear(args.d_model, args.d_model)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(args.block_size, args.block_size))
        #                              .view(1, 1, args.block_size, args.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(args.block_size + 1, args.block_size + 1))
                                     .view(1, 1, args.block_size + 1, args.block_size + 1))
        self.n_head = args.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, args):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.d_model)
        self.ln2 = nn.LayerNorm(args.d_model)
        self.attn = CausalSelfAttention(args)
        self.mlp = nn.Sequential(
            nn.Linear(args.d_model, 4 * args.d_model),
            GELU(),
            nn.Linear(4 * args.d_model, args.d_model),
            nn.Dropout(args.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mt = args.model_type
        # self.pos_emb = nn.Parameter(torch.zeros(1, args.block_size, args.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, args.block_size + 1, args.d_model))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, args.max_timestep+1, args.d_model))
        self.drop = nn.Dropout(args.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(args) for _ in range(args.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.state_emb = nn.Conv1d(in_channels=6,
                                   out_channels=args.d_model,
                                   kernel_size=3,
                                   padding=0,
                                   padding_mode='circular',
                                   bias=False)
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd),
                                     nn.Tanh())
        self.act_emb = nn.Sequential(nn.Embedding(config.vocab_size, config.d_model),
                                     nn.Tanh())
        nn.init.normal_(self.act_emb[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.args.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def argsure_optimizers(self, train_args):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_args.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_args.learning_rate, betas=train_args.betas)
        return optimizer

    # state, action, and return
    def forward(self, s, a, t=None, rtgs=None, timesteps=None):
        # s: (batch, block_size, 6)
        # a: (batch, block_size, 1)
        # t: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)


        # !!!!!---- !!!state_encoder
        state_embds = self.state_emb(s.permute(0, 2, 1)).permute(0, 2, 1)
        if a is not None and self.mt == 'reward_conditioned':
            rtg_embds = self.ret_emb(rtgs.type(torch.float32))
            act_embs = self.act_emb(a.type(torch.long).squeeze(-1)) # (batch, block_size, d_model)

            tokens = torch.zeros((s.shape[0], s.shape[1]*3 - int(t is None),
                                 self.args.d_model),
                                 dtype=torch.float32,
                                 device=state_embds.device)
            tokens[:,::3,:] = rtg_embds
            tokens[:,1::3,:] = state_embds
            tokens[:,2::3,:] = act_embs[:,-s.shape[1] + int(t is None):,:]
        elif a is None and self.mt == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embds = self.ret_emb(rtgs.type(torch.float32))
            tokens = torch.zeros((s.shape[0], s.shape[1]*2, self.args.d_model),
                                 dtype=torch.float32,
                                 device=state_embds.device)
            tokens[:,::2,:] = rtg_embds # really just [:,0,:]
            tokens[:,1::2,:] = state_embds # really just [:,1,:]
        elif a is not None and self.mt == 'naive':
            act_embs = self.act_emb(a.type(torch.long).squeeze(-1)) # (batch, block_size, d_model)
            tokens = torch.zeros((s.shape[0], s.shape[1]*2 - int(t is None),
                                 self.args.d_model),
                                 dtype=torch.float32,
                                 device=state_embds.device)
            tokens[:,::2,:] = state_embds
            tokens[:,1::2,:] = act_embs[:,-s.shape[1] + int(t is None):,:]
        elif a is None and self.mt == 'naive': # only happens at very first timestep of evaluation
            tokens = state_embds
        else:
            raise NotImplementedError()

        batch_size = s.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, d_model

        position_embds = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.args.d_model, dim=-1)) + self.pos_emb[:, :tokens.shape[1], :]

        x = self.drop(tokens + position_embds)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if a is not None and self.mt == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embds
        elif a is None and self.mt == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif a is not None and self.mt == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embds
        elif a is None and self.mt == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired t also calculate the loss
        loss = None
        if t is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1))

        return logits, loss
