"""Frozen target-centric N0..N4 models. No old trained parameters are loaded.

Construction intentionally reproduces old T1's random initialization order:
encoder, tenor identities, LN1, QKV/output, LN2, FFN, readout. The fixed mixer
uses an isolated hash-derived RNG stream after the shared backbone is built.
"""
from __future__ import annotations
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

NAMES=('N0_OwnMaskedFixed','N1_Fixed','N2_UniformCross','N3_ContentCross','N4_SelfTarget')
ALIASES={f'N{i}':name for i,name in enumerate(NAMES)}
D_MODEL=16
HEADS=2
TARGET=3
OTHERS=(0,1,2,4)


def fixed_mixer_seed(seed:int)->int:
    """Stable, platform-independent independent substream; no Python hash()."""
    raw=hashlib.sha256(f'{int(seed)}|fixed_mixer'.encode('utf-8')).digest()
    return int.from_bytes(raw[:8],'big')%(2**63)


class TargetMixer(nn.Module):
    def __init__(self, uniform:bool=False):
        super().__init__()
        self.qkv=nn.Linear(D_MODEL,3*D_MODEL)
        self.output=nn.Linear(D_MODEL,D_MODEL)
        self.uniform=uniform

    def forward(self,query:torch.Tensor,keys:torch.Tensor):
        batch,length,_=keys.shape
        q=F.linear(query,self.qkv.weight[:D_MODEL],self.qkv.bias[:D_MODEL])
        k=F.linear(keys,self.qkv.weight[D_MODEL:2*D_MODEL],self.qkv.bias[D_MODEL:2*D_MODEL])
        v=F.linear(keys,self.qkv.weight[2*D_MODEL:],self.qkv.bias[2*D_MODEL:])
        q=q.reshape(batch,1,HEADS,D_MODEL//HEADS).transpose(1,2)
        k=k.reshape(batch,length,HEADS,D_MODEL//HEADS).transpose(1,2)
        v=v.reshape(batch,length,HEADS,D_MODEL//HEADS).transpose(1,2)
        logits=q@k.transpose(-2,-1)/math.sqrt(D_MODEL//HEADS)
        weights=torch.softmax(logits*0. if self.uniform else logits,dim=-1)
        context=(weights@v).transpose(1,2).reshape(batch,1,D_MODEL)
        return self.output(context).squeeze(1),weights


class TargetCentric(nn.Module):
    def __init__(self,name:str,seed:int):
        super().__init__()
        self.name=name
        self.encoder=nn.Sequential(nn.Linear(20,D_MODEL),nn.GELU())
        self.tenor_identity=nn.Parameter(torch.empty(5,D_MODEL))
        nn.init.normal_(self.tenor_identity,mean=0.,std=.02)
        self.norm1=nn.LayerNorm(D_MODEL)
        self.mixer=TargetMixer(uniform=name=='N2_UniformCross')
        self.norm2=nn.LayerNorm(D_MODEL)
        self.ffn=nn.Sequential(nn.Linear(D_MODEL,2*D_MODEL),nn.GELU(),nn.Linear(2*D_MODEL,D_MODEL))
        self.readout=nn.Linear(D_MODEL,1)
        if name in ('N0_OwnMaskedFixed','N1_Fixed'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(fixed_mixer_seed(seed))
                self.mixer=nn.Sequential(nn.Linear(80,11),nn.GELU(),nn.Linear(11,D_MODEL))

    def encode(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=3 or tuple(x.shape[1:])!=(20,5):
            raise ValueError('x must have shape [B,20,5], in fixed tenor order 1Y/3Y/5Y/7Y/10Y')
        if self.name=='N0_OwnMaskedFixed':
            masked=torch.zeros_like(x)
            masked[:,:,TARGET]=x[:,:,TARGET]
            x=masked
        return self.encoder(x.transpose(1,2))+self.tenor_identity.unsqueeze(0)

    def forward(self,x:torch.Tensor,return_attention:bool=False):
        h=self.encode(x)
        z=self.norm1(h)
        attention=None
        if self.name in ('N0_OwnMaskedFixed','N1_Fixed'):
            context=self.mixer(z.flatten(1))
        else:
            key_indices=range(5) if self.name=='N4_SelfTarget' else OTHERS
            context,attention=self.mixer(z[:,TARGET:TARGET+1],z[:,key_indices])
        u=h[:,TARGET]+context
        v=u+self.ffn(self.norm2(u))
        logits=self.readout(v).squeeze(-1)
        return (logits,attention) if return_attention else logits


def build_model(name:str,seed:int)->TargetCentric:
    name=ALIASES.get(name,name)
    if name not in NAMES:raise ValueError(f'Unknown target-centric model: {name}')
    # No RNG side effects on the caller. All pairwise common components are
    # initialized from the same seed, exactly matching canonical old T1.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model=TargetCentric(name,int(seed))
    return model


def parameter_counts(model:TargetCentric)->dict[str,int]:
    total=sum(p.numel() for p in model.parameters())
    dead=544 if model.name=='N2_UniformCross' else 0
    expected=2652 if model.name in ('N0_OwnMaskedFixed','N1_Fixed') else 2657
    if total!=expected:raise AssertionError(f'Parameter guard failed {model.name}: {total} != {expected}')
    return {'total':total,'trainable_declared':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'structurally_zero_gradient_qk':dead,'potentially_active_excluding_uniform_qk':total-dead}
