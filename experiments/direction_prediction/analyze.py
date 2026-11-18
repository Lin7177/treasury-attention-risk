"""Paired block uncertainty for all preregistered comparisons, not winner selection."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiment import metrics
PAIRS=[('B1_multilogit','B0_ownlogit'),('T1_attention','B2_fixed'),('T1_attention','B4_uniform'),('T2_condition','T3_post_state'),('T2_condition','B3_fixed_state')]
def main(out):
    f=pd.read_csv(out/'predictions_ensemble.csv');wide=f.pivot(index='index',columns='model',values='probability_up')
    d=f.drop_duplicates('index').set_index('index').loc[wide.index].reset_index()
    assert len(d)==586 and int(d.direction.ne(0).sum())==574
    years=d.date.str[:4].to_numpy();rows=[]
    for length in [5,10,20]:
        rng=np.random.default_rng(20261104);draw={p:[] for p in PAIRS}
        for _ in range(2000):
            ix=[]
            for year in np.unique(years):
                indices=np.flatnonzero(years==year);l=min(length,len(indices));starts=rng.integers(0,len(indices)-l+1,size=int(np.ceil(len(indices)/l)))
                ix.extend(np.concatenate([indices[s:s+l] for s in starts])[:len(indices)])
            ix=np.array(ix);part=d.iloc[ix]
            for a,b in PAIRS:
                ma=metrics(wide[a].to_numpy()[ix],part);mb=metrics(wide[b].to_numpy()[ix],part)
                draw[(a,b)].append([ma['balanced_accuracy']-mb['balanced_accuracy'],ma['brier']-mb['brier']])
        for (a,b),values in draw.items():
            ma=metrics(wide[a].to_numpy(),d);mb=metrics(wide[b].to_numpy(),d)
            for j,key in enumerate(['balanced_accuracy','brier']):
                lo,hi=np.quantile(np.asarray(values)[:,j],[.025,.975]);rows.append(dict(a=a,b=b,block_length=length,metric=key,delta=ma[key]-mb[key],low=lo,high=hi,replicates=2000))
    pd.DataFrame(rows).to_csv(out/'paired_intervals.csv',index=False)
    print(pd.DataFrame(rows).query('block_length==10').to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
