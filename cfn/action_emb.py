from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class PerceiverConfig:
    dim: int
    latent_dim: int
    num_latents: int
    depth: int
    
    cross_heads: int
    cross_dim_head: int
    latent_heads: int
    latent_dim_head: int
    attn_dropout: float
    ff_dropout: float

    output_dim: int
    final_proj_head: bool


def get_sinusoid_encoding_table(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


'''
Credits to https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
'''
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

## a little modification on GEGLU()
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PerAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context = None, mask = None, return_attn = False):
        h = self.heads
        
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        
        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        
        if return_attn:
            attention_weights = attn.detach().clone()
            attention_weights = rearrange(attention_weights, '(b h) n d -> b h n d', h = h)
        
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        
        if return_attn:
            return self.to_out(out), attention_weights
        
        else:
            return self.to_out(out)

class Perceiver(nn.Module):
    def __init__(self,
                 dim,
                 latent_dim,
                 output_dim,
                 num_latents,
                 depth,
                 cross_heads = 1,
                 cross_dim_head = 64,
                 latent_heads = 8,
                 latent_dim_head = 64,
                 attn_dropout = 0.,
                 ff_dropout = 0.,
                 final_proj_head = False,
                 ) -> None:
        super().__init__()

        # self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn_blocks = nn.ModuleList([
            PreNorm(latent_dim, PerAttention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        ])

        self.layers = nn.ModuleList([])
        
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, PerAttention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout)),
                PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
            ]))

        self.proj_head = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim)
        ) if final_proj_head else nn.Identity()
        
    def forward(self, data, mask = None, return_cross_attn = False):
        b = data.shape[0]
        
        x = repeat(self.latents, 'n d -> b n d', b = b)
        
        cross_attn, cross_ff = self.cross_attn_blocks

        # cross attention only happens once for Perceiver IO
        if not return_cross_attn:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x
            
        else:
            outputs = cross_attn(x, context = data, mask = mask, return_attn = return_cross_attn)
            x = x + outputs[0]
            x = cross_ff(x) + x

        # layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if return_cross_attn:
            return self.proj_head(x), outputs[1]
        
        else:
            return self.proj_head(x)
        
### Encoding actions of different agents
class SequentialActionEmb(nn.Module):
    def __init__(self,
                 num_agents: int, # agent num
                 num_steps_conditioning: int, # ?
                 action_dim: int, 
                 is_continuous_act: bool,# 是否连续action
                 perceiver_cfg: PerceiverConfig,
                 ):
        super().__init__()
        
        self.perceiver_cfg = perceiver_cfg

        # 将动作映射到emb
        if is_continuous_act:
            self.token_emb = nn.Sequential(
                nn.Linear(action_dim, perceiver_cfg.dim),
                nn.SiLU(),
                nn.Linear(perceiver_cfg.dim, perceiver_cfg.dim)
            )
        else:
            self.token_emb = nn.Embedding(action_dim, perceiver_cfg.dim)

        ### 依赖timestep emb和agent emb就足以区分不同timestep下相同agent的action feat或者相同timestep下不同agent的action feat
        # timestep_emb是表现time step的
        self.timestep_emb = nn.Embedding(num_steps_conditioning, perceiver_cfg.dim)

        # Option 1
        ## Learnable Agent Emb
        # 对每个agent 添加唯一标识量的动作
        self.agent_emb = nn.Embedding(num_agents, perceiver_cfg.dim)
        # Option 2
        ## Unlearnable Agent Emb
        # self.agent_emb = get_sinusoid_encoding_table(num_agents, perceiver_cfg.dim)

        # 核心注意力机制
        self.perceiver_io = Perceiver(**perceiver_cfg.__dict__)

    def forward(self, x, mask = None, return_cross_attn = False):
        # 输入x.shape = [B, T, N, action_dim]
        # mask.shape = [B, T, N]
        # btn 分别是 batch, time_step, agent num
        b, t = x.shape[:2]
        device = x.device

        # emb化 action, 输出变成了 shape = [B, T, N, D]# 让模型知道是哪个agent做的动作
        x = self.token_emb(x)

        # 让模型知道是哪个agent做的动作
        # agent_emb = self.agent_emb(torch.arange(n, device = device)) # [N, D]
        # agent_emb = rearrange(agent_emb, 'n d -> () n d')
        # x = rearrange(x, 'b t n d -> (b t) n d') # [B T N D] -> [B*T, N, D]
        # x = x + agent_emb
        # x = rearrange(x, '(b t) n d -> b t n d', b=b, t=t)

        t_emb = self.timestep_emb(torch.arange(t, device = device))
        t_emb = rearrange(t_emb, 't d -> () t d')
        # x = rearrange(x, 'b t n d -> (b n) t d')
        x = x + t_emb
        # x = rearrange(x, '(b n) t d -> b t n d', b=b, n=n)

        # x = rearrange(x, 'b t n d -> b (t n) d')
        mask = torch.ones(b, t, dtype=torch.bool).to(x.device)

        if return_cross_attn:
            act_cond, cross_attn = self.perceiver_io(x, mask = mask, return_cross_attn = return_cross_attn)
        else:
            act_cond = self.perceiver_io(x, mask = mask, return_cross_attn = return_cross_attn)
            cross_attn = None

        return act_cond, cross_attn
        


# simple joint action encoder
class SimpleActionEncoder(nn.Module):
    def __init__(self,
                 num_agents: int,
                 action_dim: int,
                 is_continuous_act: bool,
                 embed_dim: int,
                 output_dim: int,
                 num_heads: int,
                 attn_dropout: float,
                 ff_dropout: float,
                 depth: int,
                 ):
        super().__init__()

        if is_continuous_act:
            self.token_emb = nn.Sequential(
                nn.Linear(action_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.token_emb = nn.Embedding(action_dim, embed_dim)
            
        self.pos_emb = nn.Embedding(num_agents, embed_dim)

        self.layers = nn.ModuleList([])
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, PerAttention(embed_dim, heads = self.num_heads, dim_head = self.dim_head, dropout = attn_dropout)),
                PreNorm(embed_dim, FeedForward(embed_dim, dropout = ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        '''
        shape of x should be (B, N, act_dim) or (B, N,)
        '''
        b, n = x.shape[:2]
        device = x.device

        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb
        
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return self.to_out(x)
